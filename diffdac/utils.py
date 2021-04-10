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

Taken from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
and slightly modified.
"""
import os
import glob
from collections import OrderedDict
from datetime import datetime

import torch.nn as nn

from diffdac.envs import VecNormalize


def get_render_func(venv):
    """ Returns the correct render function for a possibly vectorized environment. """
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    """ Returns a vector normalized version of a (possibly) vectorised environment. """
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for KFAC implementation.
class AddBias(nn.Module):
    """ Adds a bias offset to a tensor """
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def cleanup_log_dir(log_dir):
    """ Removes log files from a log directory if it exists and creates it if not """
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def cleanup_save_dir(save_dir):
    """ Removes parameter save files from a save directory if it exists and creates it if not """
    try:
        os.makedirs(save_dir)
    except OSError:
        files = glob.glob(os.path.join(save_dir, '*.pt'))
        for f in files:
            os.remove(f)


def get_args_string(args):
    """
    Creates a string summarising the argparse arguments.
    :param args: parser.parse_args()

    :return: String of the arguments of the argparse namespace.
    """
    string = ''
    if hasattr(args, 'experiment_name'):
        string += f'{args.experiment_name} ({datetime.now()})\n'
    max_length = max([len(k) for k, _ in vars(args).items()])
    new_dict = OrderedDict((k, v) for k, v in sorted(
        vars(args).items(), key=lambda x: x[0]))
    for key, value in new_dict.items():
        string += ' ' * (max_length - len(key)) + key + ': ' + str(value) + '\n'
    return string