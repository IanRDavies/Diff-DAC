# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import ast
import argparse
import numpy as np

import torch


def get_args(arg_dict=None):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--sync-freq',
        type=int,
        default=0,
        help='max amount of message staleness for local parameter diffusion'
    )
    parser.add_argument(
        '--num-learners',
        type=int,
        default=25,
        help='number of learners'
    )
    parser.add_argument(
        '--env-group-spec',
        type=str,
        default='None',
        help='A 2-tuple helping to define the environments for each agent. '
             'The tuple will be structured as (number of groups, envs per group). '
             'Environments are the same (except the seed) within groups and differ between groups.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=7e-4,
        help='learning rate (default: 7e-4)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)'
    )
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)'
    )
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)'
    )
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=889,
        help='random seed'
    )
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)"
    )
    parser.add_argument(
        '--num-procs-per-learner',
        type=int,
        default=1,
        help='num environment simulators per learner (default: 1)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=int(10e3),
        help='max episode length (default: 10,000)'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=None,
        help='maximum gradient norm used in gradient clipping (if None, do not clip)'
    )
    parser.add_argument(
        '--num-steps-per-update',
        type=int,
        default=5,
        help='number of forward steps in the environment before parameter updates (default: 5)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, measured in environment steps (default: 10)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100000,
        help='Interval between parameter save points in environment steps (default: 100000)'
    )
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e6,
        help='number of total environment steps to train (default: 10e6)'
    )
    parser.add_argument(
        '--env-name',
        default='RandomExtremeAcrobot-v1',
        help='environment to train on (default: RandomExtremeAcrobot-v1)'
    )
    parser.add_argument(
        '--log-dir',
        default='/tmp/logging/acrobot',
        help='directory to save agent logs'
    )
    parser.add_argument(
        '--cuda-device',
        type=int,
        default=0,
        help='index of cuda device to use'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits'
    )
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy'
    )
    parser.add_argument(
        '--adjacency-matrix',
        default="""[
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        ]""",
        type=str,
        help='Adjacency matrix which is provided is used to build the agent graph.'
    )
    parser.add_argument(
        '--link-drop-proportion',
        default=0.0,
        type=float,
        help='The probability with which messages are not passed at each update.'
             'This probability parameterises an independent Bernoulli distribution for each agent.'
    )
    parser.add_argument(
        "--separate-intialisation",
        default=False,
        action="store_true",
        help="Initialise all agents to the differing points in parameter space."
    )
    parser.add_argument("--server-mode", default=False, action="store_true")

    args = parser.parse_args(arg_dict)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.adjacency_matrix = np.array(ast.literal_eval(args.adjacency_matrix))
    # Set the logging folder from the load automatically.
    args.eval_log_dir = os.path.join(args.log_dir, 'eval')
    args.save_dir = os.path.join(args.log_dir, 'trained_models')
    args.env_group_spec = ast.literal_eval(args.env_group_spec)

    if args.env_group_spec:
        assert (isinstance(args.env_group_spec, tuple) and len(args.env_group_spec) == 2
                and np.prod(args.env_group_spec) == args.num_procs_per_learner), \
            'The environment list config is misspecified. Expected a 2-tuple and ' \
            f'got {args.env_group_spec}'

    return args
