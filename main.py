# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pprint as pp
from collections import deque
from copy import deepcopy

import torch
import torch.multiprocessing as mp

from pyvirtualdisplay import Display

from diffdac.arguments import get_args
from diffdac.storage import RolloutStorage
from diffdac.model import Policy
from diffdac.gpu_gossip_buffer import GossipBuffer
from diffdac.diffdac_a2c import DiffDAC_A2C
from diffdac.graph_manager import SymmetricConnectionGraph
from diffdac.utils import get_args_string, cleanup_log_dir, cleanup_save_dir


def actor_learner(args, rank, barrier, device, gossip_buffer, env_group_spec=None):
    """ Single Actor-Learner Process """
    if args.server_mode:
        Display(visible=0, size=(1400, 900)).start()

    # Set random seeds
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(device)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # (Hack) Import here to ensure OpenAI-gym envs only run on the CPUs
    # corresponding to the processes' affinity
    from diffdac.envs import make_vec_envs
    # Make envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_procs_per_learner,
                         args.gamma, args.log_dir, device, False, rank=rank,
                         env_group_spec=env_group_spec, signature=f'Agent-Rank-{rank}')

    # Policy Setup
    base_kwargs = {'recurrent': args.recurrent_policy}
    base_kwargs['init_repeats'] = rank if args.separate_intialisation else 1

    # Initialize actor_critic
    actor_critic = Policy(
        env_name=args.env_name,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        base_kwargs=base_kwargs)
    actor_critic.to(device)

    # Initialize agent
    agent = DiffDAC_A2C(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm,
        rank=rank,
        gossip_buffer=gossip_buffer,
        link_drop_prob=args.link_drop_proportion
    )

    rollouts = RolloutStorage(args.num_steps_per_update,
                              args.num_procs_per_learner,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # Synchronize agents before starting training
    barrier.wait()
    print('%s: barrier passed' % rank)

    # Start training

    # Calculate training intervals in terms of parameter updates rather than environment steps.
    num_updates = int(args.num_env_steps) // (
        args.num_steps_per_update * args.num_procs_per_learner * args.num_learners
    )
    save_interval = int(args.save_interval) // (
            args.num_steps_per_update * args.num_procs_per_learner * args.num_learners
    )

    for j in range(num_updates):
        # Make sure we save parameters close to the end of training even if agents are out of sync.
        to_go = num_updates - j
        if to_go < args.sync_freq:
            torch.save(
                [actor_critic.state_dict()],
                os.path.join(args.save_dir, f'params_{to_go}_to_go_agent_{rank}.pt')
            )

        # Step through environment
        for step in range(args.num_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        # --/

        # Update parameters
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]
            ).detach()
        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits
        )
        agent.update(rollouts)
        rollouts.after_update()
        # --/

        # Save every "save_interval" local environment steps (or last update)
        if (j % save_interval == 0 or j == num_updates - 1) and args.save_dir != '':
            torch.save(
                [actor_critic.state_dict()],
                os.path.join(args.save_dir, '%s.%.3d.pt' % (rank, j // save_interval)))
        # --/

    # Save agent parameters at the end of training.
    torch.save([actor_critic.state_dict()], os.path.join(args.save_dir, f'final_params_{rank}.pt'))


def make_gossip_buffer(args, mng, device):
    """ Builds a gossip buffer shared by all agents for the purpose of message passing """
    # Make local-gossip-buffer
    if args.num_learners > 1:
        # Make Topology
        topology = []
        for rank in range(args.num_learners):
            graph = SymmetricConnectionGraph(args.adjacency_matrix, rank)
            topology.append(graph)

        # Initialize "actor_critic-shaped" parameter-buffer
        actor_critic = Policy(
                env_name=args.env_name,
                base_kwargs={'recurrent': args.recurrent_policy}
            )
        
        actor_critic.to(device)

        # Keep track of local iterations since learner's last sync
        sync_list = mng.list([0 for _ in range(args.num_learners)])
        # Used to ensure proc-safe access to agents' message-buffers
        buffer_locks = mng.list([mng.Lock() for _ in range(args.num_learners)])
        # Used to signal between processes that message was read
        read_events = mng.list([
            mng.list([mng.Event() for _ in range(args.num_learners)])
            for _ in range(args.num_learners)
        ])
        # Used to signal between processes that message was written
        write_events = mng.list([
            mng.list([mng.Event() for _ in range(args.num_learners)])
            for _ in range(args.num_learners)])

        # Need to maintain a reference to all objects in main processes
        _references = [topology, actor_critic, buffer_locks,
                       read_events, write_events, sync_list]
        
        gossip_buffer = GossipBuffer(topology, actor_critic, buffer_locks,
                                        read_events, write_events, sync_list,
                                        sync_freq=args.sync_freq)
    else:
        # Only one agent so no gossip buffer needed
        _references = None
        gossip_buffer = None

    return gossip_buffer, _references


def train(args):
    """ The main function which sets up and trains agents. """
    pp.pprint(args)

    proc_manager = mp.Manager()
    barrier = proc_manager.Barrier(args.num_learners)

    # Shared-gossip-buffer on GPU-0 or CPU if no cuda.
    device = torch.device('cuda:%s' % 0 if args.cuda else 'cpu')
    shared_gossip_buffer, _references = make_gossip_buffer(args, proc_manager, device)

    # Make actor-learner processes (one per learner)
    proc_list = []
    for rank in range(args.num_learners):

        proc = mp.Process(
            target=actor_learner,
            args=(args, rank, barrier, device, shared_gossip_buffer, args.env_group_spec),
            daemon=False
        )
        proc.start()
        proc_list.append(proc)

        # # Bind agents to specific hardware-threads (generally not necessary)
        # avail = list(os.sched_getaffinity(proc.pid))  # available-hwthrds
        # cpal = math.ceil(len(avail) / args.num_learners)  # cores-per-proc
        # mask = [avail[(rank * cpal + i) % len(avail)] for i in range(cpal)]
        # print('process-mask:', mask)
        # os.sched_setaffinity(proc.pid, mask)

    for proc in proc_list:
        proc.join()


if __name__ == "__main__":
    # Set up
    mp.set_start_method('forkserver')
    torch.set_num_threads(1)
    args = get_args()

    if args.server_mode:
        # In cases where the environments do not run without some screen.
        Display(visible=0, size=(1400, 900)).start()

    # Make/clean save & log directories
    cleanup_log_dir(args.log_dir)
    cleanup_save_dir(args.save_dir)
    # Save run parameters for posterity
    with open(os.path.join(args.log_dir, 'params.txt'), 'w') as f:
        f.write(get_args_string(args))

    train(args)
    print('TRAINING COMPLETE')
