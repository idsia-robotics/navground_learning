from __future__ import annotations

import numpy as np

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv
    import gymnasium as gym


def maybe_make_venv(
    env: gym.Env[Any, Any] | VecEnv,
    parallel: bool = False,
    num_envs: int = 1,
    rng: np.random.Generator = np.random.default_rng(0)
) -> VecEnv:
    from stable_baselines3.common.vec_env import VecEnv
    if isinstance(env, VecEnv):
        return env
    return make_vec_from_env(env, parallel, num_envs, rng)


def make_vec_from_env(
    env: gym.Env[Any, Any],
    parallel: bool = False,
    num_envs: int = 1,
    rng: np.random.Generator = np.random.default_rng(0)
) -> VecEnv:
    """
    Creates an imitation-compatible vectorized enviroment.
    Just a tiny wrapped on :py:func:`imitation.util.util.make_vec_env`
    that reads ``env_id`` and  ``env_make_kwargs`` from the env.

    :param      env:       The environment
    :param      parallel:  Whether to run the venv in parallel
    :param      number:    The number of environments
    :param      rng:       The random number generator

    :returns:   The vectorized environment.
    """
    from imitation.util.util import make_vec_env
    from imitation.data.wrappers import RolloutInfoWrapper

    if not env.unwrapped.spec:
        raise ValueError("env has an empty spec")
    env_id = env.unwrapped.spec.id
    env_kwargs = env.unwrapped.spec.kwargs
    return make_vec_env(env_id,
                        rng=rng,
                        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                        env_make_kwargs=env_kwargs,
                        parallel=parallel,
                        n_envs=num_envs)
