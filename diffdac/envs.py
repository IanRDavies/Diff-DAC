"""
MIT License

Copyright (c) 2017 Ilya Kostrikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""

# Monkey patch baselines types dictionary to expand it as a quickfix
import numpy as np
from ctypes import c_double
from baselines.common.vec_env.shmem_vec_env import _NP_TO_CT
_NP_TO_CT[np.float64] = c_double


import os

import gym
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from envs.register import register_custom_envs
register_custom_envs()

try:
    import dm_control2gym
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets, signature='', max_steps=None,
             heterogeneous=False):
    """ Returns function which sets up the required environment. """
    def _thunk():
        env = gym.make(env_id)
        # Copy the seed value to a version which may be updated to create log file names.
        s = seed

        # This if catches all the multi-task envs used in experiments
        if 'mujoco' in env_id.lower() or env_id == 'RandomExtremeAcrobot-v1':
            np.random.seed(seed + rank)
            env.seed(seed + rank)
            # Account for cases where environments are to be reset each episode using their own seed
            # but where environment parameters (in the multi-task setting) are identical and where
            # both initial state and environment parameter sampling are independent
            # (i.e. where heterogeneous=True)
            if heterogeneous:
                env.sample_parameters(seed + rank)
                s = seed + rank
            else:
                env.sample_parameters(seed)
                s = seed

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            # Set up logging with file names that also copture environment parameters where relevant
            if 'mujoco' in env_id.lower():
                to_add = f'Wind: {env.model.opt.wind}\tMass: {env.model.body_mass}'
            elif env_id == "RandomExtremeAcrobot-v1":
                to_add = f'm-{env.mass}-l-{env.length}-i-{env.inertia}'
            else:
                to_add = ''
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank) + signature + f"seed-{s}" + to_add),
                allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device,
                  allow_early_resets, num_frame_stack=None, rank=0,
                  signature='', max_steps=None, env_group_spec=None):
    """ Make vectorised environments for parallelized experience sampling. """
    # Should environments be the all the same for each learner or differ across processes for the
    # same learner.
    heterogeneous_envs = not (env_group_spec is not None and env_group_spec[1] == num_processes)
    if env_group_spec is None or env_group_spec[0] == 1:
        # No grouping of environment processes for each agent.
        envs = [
            make_env(env_name, seed + num_processes * rank, (rank * num_processes) + i,
                     log_dir, allow_early_resets, signature, max_steps,
                     heterogeneous=heterogeneous_envs)
            for i in range(num_processes)
        ]
    else:
        # We have environments grouped such that environments differ even for the same learner.
        envs = []
        counter = 0
        for i in range(env_group_spec[0]):
            envs += [
                make_env(env_name, seed + num_processes * rank,
                         (rank * num_processes) + counter + i, log_dir,
                         allow_early_resets, signature, max_steps, heterogeneous=False)
                for i in range(env_group_spec[1])
            ]
            seed += env_group_spec[1]
            counter += env_group_spec[1]

    # Allow dummy environment wrapper if no parallelisation required.
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    # Ensure environments are compatible with the PyTorch agents.
    envs = VecPyTorch(envs, device)

    # Frame stacking for visual environments.
    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class VecPyTorch(VecEnvWrapper):
    """ Ensures vectorised environments are compatible with PyTorch agents """
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    """ Clips and normalises observation vectors. """
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

