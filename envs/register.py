from gym.envs.registration import register


def register_custom_envs():
    """ Registers custom environments with gym (so we can use `gym.make`) """
    register(
        id="RandomMuJoCoHopper-v2",
        entry_point="envs.mujoco:make_random_hopper"
    )
    register(
        id="RandomMuJoCoHalfCheetah-v2",
        entry_point="envs.mujoco:make_random_half_cheetah"
    )
    register(
        id="MuJoCoHopper-v2",
        entry_point="envs.mujoco:make_hopper_with_air"
    )
    register(
        id="MuJoCoHalfCheetah-v2",
        entry_point="envs.mujoco:make_half_cheetah_with_air"
    )
    register(
        id="RandomWindMuJoCoHalfCheetah-v2",
        entry_point="envs.mujoco:make_random_wind_half_cheetah"
    )
    register(
        id="RandomMassMuJoCoHalfCheetah-v2",
        entry_point="envs.mujoco:make_random_mass_half_cheetah"
    )
    register(
        id="RandomWindMuJoCoHopper-v2",
        entry_point="envs.mujoco:make_random_wind_hopper"
    )
    register(
        id="RandomMassMuJoCoHopper-v2",
        entry_point="envs.mujoco:make_random_mass_hopper"
    )
    register(
        id="RandomNormalAcrobot-v1",
        entry_point='envs.acrobot:RandomNormalAcrobot',
        max_episode_steps=500
    )
    register(
        id="RandomExtremeAcrobot-v1",
        entry_point='envs.acrobot:RandomExtremeAcrobot',
        max_episode_steps=500
    )
