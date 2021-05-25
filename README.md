# Diff-DAC: Fully Distributed Actor-Critic Architecture for Multitask Deep Reinforcement Learning
This repo contains the code used in developing the DiffDAC architecture for distributed multi-task reinforcement learning as described in the paper by [Sergio Valcarcel Macua](https://github.com/sergiovalmac), [Ian Davies](https://github.com/IanRDavies), Aleksi Tukiainen and Enrique Munoz de Cote \[1\].

Below we share the abstract from the paper to provide an overview:

> We propose a fully distributed actor-critic architecture, named Diff-DAC, with application to multitask reinforcement learning (MRL). During the learning process, agents communicate their value and policy parameters to their neighbours, diffusing the information across a network of agents with no need for a central station. Each agent can only access data from its local task, but aims to learn a common policy that performs well for the whole set of tasks. The architecture is scalable, since the computational and communication cost per agent depends on the number of neighbours rather than the overall number of agents.We derive Diff-DAC from duality theory and provide novel insights into the actor-critic framework, showing that it is actually an instance of the dual ascent method.We prove almost sure convergence of Diff-DAC to a common policy under general assumptions that hold even for deep-neural network approximations. For more restrictive assumptions, we also prove that this common policy is a stationary point of an approximation of the original problem. Numerical results on multitask extensions of common continuous control benchmarks demonstrate that Diff-DAC stabilises learning and has a regularising effect that induces higher performance and better generalisation properties than previous architectures.

\[1\] Valcarcel Macua, S., Davies, I., Tukiainen, A., & De Cote, E. (2021). Fully distributed actor-critic architecture for multitask deep reinforcement learning. The Knowledge Engineering Review, 36, E6. doi:10.1017/S0269888921000023

## Setup
This code has been tested with
* Python 3.7.4
* PyTorch 1.0 or higher
The experiments reported in the paper were run using PyTorch 1.5 (The original GALA Paper uses PyTorch 1.0).
We have also tested this code with PyTorch 1.3.

### Requirements
To install other requirements, return to the clone this repo and run
```
pip install -r requirements.txt
```

If looking to run on MuJoCo environments then install MuJoCo and `mujoco-py` following the instructions
[here](https://github.com/openai/mujoco-py).

### Running the code
As an example, to use Diff-A2C to train an agent to play the multitask variant of the `Acrobot-v1`
environment using 4 actor-learners arranged in a ring with 1 simulator per actor-learner, run
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env-name RandomExtremeAcrobot-v1 \
    --seed 1 \
    --lr 0.00007 \
    --num-env-steps 1000000 \
    --save-interval 50000 \
    --num-learners 4 \
    --adjacency-matrix "[[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]"
    --num-steps-per-update 5 \
    --sync-freq 5 \
    --num-procs-per-learner 1 \
    --log-dir /tmp/logs/Acrobot/
```

This code produces one log file for each simulator.
The log file contains three columns, the reward, episode length, and wall clock time, recorded after every episode.

## Environments
The multi-task Acrobot variant used in our experiments is a copy of the implementation used by
[Packer et al. (2018)](https://arxiv.org/abs/1810.12282) which is available [here](https://github.com/sunblaze-ucb/rl-generalization) (alongside other variants of reinforcement learning
 environments) and is absorbed into our codebase for simplicity.

We also implement wrappers to enable variants of standard MuJoCo environments in [./envs/mujoco.py](./envs/mujoco.py).


## Acknowledgements
**Gossip-based Actor-Learner Architectures (GALA)**

This repo was built upon the [implementation of GALA](https://github.com/facebookresearch/gala)
used for the experiments reported in
> Mido Assran, Joshua Romoff, Nicolas Ballas, Joelle Pineau, and Mike Rabbat, "Gossip-based actor learner architectures for deep reinforcement learning," *Advances in Neural Information Processing Systems (NeurIPS)* 2019. [arxiv version](https://arxiv.org/abs/1906.04585)

Their code is itself based on [Ilya Kostrikov's](https://github.com/ikostrikov) pytorch-a2c-ppo-acktr-gail [repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)


## License
See the LICENSE file for details about the license under which this code is made available.

