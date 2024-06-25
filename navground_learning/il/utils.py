import multiprocessing as mp
from collections import ChainMap
from collections.abc import Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pettingzoo as pz
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from ..env import NavgroundBaseEnv, NavgroundEnv
from .pz_utils import make_ma_venv


def make_venv(env: gym.Env | pz.ParallelEnv | None = None,
              parallel: bool = False,
              n_envs: int = 8,
              rng=np.random.default_rng(0),
              **kwargs: Any) -> tuple[VecEnv, NavgroundBaseEnv]:
    if parallel:
        processes = mp.cpu_count()
    else:
        processes = 1
    if isinstance(env, pz.ParallelEnv):
        return make_ma_venv(env, processes=processes, n_envs=n_envs, **kwargs)
    if isinstance(env, gym.Env):
        return make_sa_venv(env,
                            parallel=parallel,
                            n_envs=n_envs,
                            rng=rng,
                            **kwargs)
    if 'agent_index' in kwargs:
        return make_sa_venv(env=None,
                            parallel=parallel,
                            n_envs=n_envs,
                            rng=rng,
                            **kwargs)
    return make_ma_venv(env, processes=processes, n_envs=n_envs, **kwargs)


def make_sa_venv(env: gym.Env | None = None,
                 parallel: bool = False,
                 n_envs: int = 8,
                 rng=np.random.default_rng(0),
                 **kwargs: Any) -> tuple[VecEnv, NavgroundEnv]:

    env_id = "navground"
    env_kwargs: Mapping[str, Any] = kwargs
    if env is not None:

        if hasattr(env, 'num_envs'):
            # is a venv
            if isinstance(env, VecEnv):
                return env, env.get_attr(
                    "config")[0], *env.get_attr("get_behavior_and_sensor")[0]()
            else:
                raise ValueError("Provide a SB3 vector env")

        if env.unwrapped.spec:
            env_id = env.unwrapped.spec.id
            env_kwargs = ChainMap(kwargs, env.unwrapped.spec.kwargs)
        else:
            if isinstance(env.unwrapped, NavgroundEnv):
                env_kwargs = env.unwrapped.init_args
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
    return venv, cast(NavgroundEnv, env.unwrapped)
