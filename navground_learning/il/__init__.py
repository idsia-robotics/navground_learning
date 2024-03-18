from collections import ChainMap
from typing import Any, Mapping, Tuple

import gymnasium as gym
import numpy as np
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from navground import core, sim
from stable_baselines3.common.vec_env import VecEnv

from ..env import NavgroundEnv
from ..utils import GymAgentConfig


def make_imitation_venv(
        env: gym.Env | None = None,
        parallel: bool = False,
        n_envs: int = 8,
        rng=np.random.default_rng(0),
        **kwargs: Any
) -> Tuple[VecEnv, GymAgentConfig, core.Behavior, sim.Sensor]:

    env_id = "navground"
    env_kwargs: Mapping[str, Any] = kwargs
    if env is not None:

        if hasattr(env, 'num_envs'):
            # is a venv
            if isinstance(env, VecEnv):
                return env, env.get_attr("config")[0], *env.get_attr("get_behavior_and_sensor")[0]()
            else:
                raise ValueError("Provide a SB3 vector env")

        if env.spec:
            env_id = env.unwrapped.spec.id
            env_kwargs = ChainMap(kwargs, env.spec.kwargs)
        else:
            if isinstance(env.unwrapped, NavgroundEnv):
                env_kwargs = env.get_init_args()
            else:
                raise ValueError(f"{env} is not a valid")

    else:
        env = gym.make("navground", **env_kwargs)

    venv = make_vec_env(env_id,
                        rng=rng,
                        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                        env_make_kwargs=env_kwargs,
                        parallel=parallel,
                        n_envs=n_envs)

    config = env.unwrapped.config
    behavior, sensor = env.unwrapped.get_behavior_and_sensor()

    return venv, config, behavior, sensor
