import gym
import numpy as np


class RandomisedMuJoCo(gym.Wrapper):
    """ A gym environment wrapper that randomises parameters for MuJoCo Environments. """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def sample_parameters(self, seed: int):
        """ Sample the environments parameters for wind and scaling mass and apply them """
        random = np.random.RandomState(seed)

        # Wind
        # Changing viscosity and density of medium from 0 to that of air so wind has an effect.
        self.model.opt.viscosity = 0.0002
        self.model.opt.density = 1.2
        # We consider wind in only the xy plane.
        self.model.opt.wind[:2] = 30 * (random.beta(0.4, 0.4, size=2) - 0.5)

        # Mass
        self.model.body_mass[:] = self.model.body_mass * (1 + 1.9 * (random.beta(0.1, 0.1) - 0.5))

        self.sim.forward()


class MuJoCoWithAir(gym.Wrapper):
    """
    A gym environment wrapping a MuJoCo environment to make the medium have the properties of air.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.model.opt.viscosity = 0.0002
        self.model.opt.density = 1.2
        self.sim.forward()

    def sample_parameters(self, *args):
        pass


class RandomisedMassMuJoCo(gym.Wrapper):
    """ A gym environment wrapper which randomises the agent's mass in a MuJoCo environment """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def sample_parameters(self, seed: int):
        """ Sample the environment's parameters for scaling the agent's mass and apply them """
        random = np.random.RandomState(seed)

        # Changing viscosity and density of medium from 0 to that of air.
        self.model.opt.viscosity = 0.0002
        self.model.opt.density = 1.2
        # Mass
        self.model.body_mass[:] = self.model.body_mass * (1 + 1.9 * (random.beta(0.1, 0.1) - 0.5))
        self.sim.forward()


class RandomisedWindMuJoCo(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def sample_parameters(self, seed: int):
        """ Sample the environment's parameters for wind in the xy plane and apply them """
        random = np.random.RandomState(seed)

        # Wind
        # Changing viscosity and density of medium from 0 to that of air so wind has an effect.
        self.model.opt.viscosity = 0.0002
        self.model.opt.density = 1.2
        # We consider wind in only the xy plane.
        wind = 30 * (random.beta(0.4, 0.4, size=2) - 0.5)
        self.model.opt.wind[:2] = wind

        self.sim.forward()
        return wind


# Environment buiding functions used to easily register environments in `register.py`

def make_random_hopper():
    standard_hopper = gym.make('Hopper-v2')
    return RandomisedMuJoCo(standard_hopper)


def make_random_half_cheetah():
    standard_half_cheetah = gym.make('HalfCheetah-v2')
    return RandomisedMuJoCo(standard_half_cheetah)


def make_hopper_with_air():
    standard_hopper = gym.make('Hopper-v2')
    return MuJoCoWithAir(standard_hopper)


def make_half_cheetah_with_air():
    standard_half_cheetah = gym.make('HalfCheetah-v2')
    return MuJoCoWithAir(standard_half_cheetah)


def make_random_wind_hopper():
    standard_hopper = gym.make('Hopper-v2')
    return RandomisedWindMuJoCo(standard_hopper)


def make_random_mass_hopper():
    standard_hopper = gym.make('Hopper-v2')
    return RandomisedMassMuJoCo(standard_hopper)


def make_random_wind_half_cheetah():
    standard_half_cheetah = gym.make('HalfCheetah-v2')
    return RandomisedWindMuJoCo(standard_half_cheetah)


def make_random_mass_half_cheetah():
    standard_half_cheetah = gym.make('HalfCheetah-v2')
    return RandomisedMassMuJoCo(standard_half_cheetah)
