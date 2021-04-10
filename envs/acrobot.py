# Acrobot Variants Copied from SunBlaze RL Generalisation
# https://github.com/sunblaze-ucb/rl-generalization

import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv


def uniform_exclude_inner(np_uniform, a, b, a_i, b_i):
    """Draw sample from uniform distribution, excluding an inner range"""
    if not (a < a_i and b_i < b):
        raise ValueError(
            "Bad range, inner: ({},{}), outer: ({},{})".format(a_i, b_i, a, b))
    while True:
        # Resample until value is in-range
        _range = a_i - a + b - b_i
        value = np_uniform(a, a + _range)
        if value < a_i:
            return value
        else:
            return value + b_i - a_i


class ModifiableAcrobotEnv(AcrobotEnv):
    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_LENGTH = 0.75
    RANDOM_UPPER_LENGTH = 1.25
    EXTREME_LOWER_LENGTH = 0.5
    EXTREME_UPPER_LENGTH = 1.5

    RANDOM_LOWER_INERTIA = 0.75
    RANDOM_UPPER_INERTIA = 1.25
    EXTREME_LOWER_INERTIA = 0.5
    EXTREME_UPPER_INERTIA = 1.5

    def reset(self, new=True):
        self.nsteps = 0
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        return super(ModifiableAcrobotEnv, self)._get_ob()

    @property
    def parameters(self):
        return {'id': self.spec.id, }

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""

        self.nsteps += 1
        x = self.np_random
        self.np_random = np.random
        if not isinstance(args[0], int):
            args = (int(args[0]), *args[1:])
        ret = super().step(*args, **kwargs)
        self.np_random = x

        # Moved logic to step wrapper because success triggers done which
        # triggers reset() in a higher level step wrapper
        # With logic in is_success(),
        # we need to cache the 'done' flag ourselves to use in is_success(),
        # since the wrapper around this wrapper will call reset immediately

        target = 90
        if self.nsteps <= target and self._terminal():
            # print("[SUCCESS]: nsteps is {}, reached done in target {}".format(
            #      self.nsteps, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, step limit {}".format(
            #      self.nsteps, target))
            self.success = False

        return ret

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: swing the end of the second link to the desired height within
        90 time steps
        """
        return self.success


class RandomNormalAcrobot(ModifiableAcrobotEnv):

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self):
        super(RandomNormalAcrobot, self).__init__()
        self.sample_parameters()

    def sample_parameters(self):
        self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.inertia = self.np_random.uniform(self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(self, new=True):
        if new:
            self.sample_parameters()
        # reset just resets .state
        return super(RandomNormalAcrobot, self).reset()

    @property
    def parameters(self):
        parameters = super(RandomNormalAcrobot, self).parameters
        parameters.update({'mass': self.mass, 'length': self.length, 'inertia': self.inertia, })
        return parameters


class RandomExtremeAcrobot(ModifiableAcrobotEnv):

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self):
        super(RandomExtremeAcrobot, self).__init__()
        self.sample_parameters()

    def sample_parameters(self, seed=None):
        if seed is None:
            np_random = self.np_random
        else:
            np_random = np.random.RandomState(seed)
        self.mass = uniform_exclude_inner(np_random.uniform,
                                          self.EXTREME_LOWER_MASS, self.EXTREME_UPPER_MASS,
                                          self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.length = uniform_exclude_inner(np_random.uniform,
                                            self.EXTREME_LOWER_LENGTH, self.EXTREME_UPPER_LENGTH,
                                            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.inertia = uniform_exclude_inner(np_random.uniform,
                                             self.EXTREME_LOWER_INERTIA, self.EXTREME_UPPER_INERTIA,
                                             self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(self, new=False):
        if new:
            self.sample_parameters()
        # reset just resets .state
        return super(RandomExtremeAcrobot, self).reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeAcrobot, self).parameters
        parameters.update({'mass': self.mass, 'length': self.length, 'inertia': self.inertia, })
        return parameters
