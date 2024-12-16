from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pettingzoo.utils.env import ParallelEnv
    from stable_baselines3.common.vec_env import VecEnv

    from ..types import Action, Observation


def make_vec_from_penv(env: ParallelEnv[int, Observation, Action],
                       num_envs: int = 1,
                       processes: int = 1) -> VecEnv:
    """
    Creates an imitation-compatible vectorized enviroment from
    a :py:class:`PettingZoo  Parallel environment <pettingzoo.utils.env.ParallelEnv>`,
    similarly as :py:func:`navground.learning.parallel_env.make_vec_from_penv` but
    adding a (custom) ``RolloutInfoWrapper`` to collect rollouts.

    It first creates a :py:class:`supersuit.vector.MarkovVectorEnv`, and then
    applies :py:func:`supersuit.concat_vec_envs_v1` to concatenate ``number`` copies of it.

    :param      env:       The environment
    :param      num_envs:  The number of pettingzoo environments to stuck together
    :param      processes: The number of (parallel) processes

    :returns:   The vectorized environment.
    """
    import supersuit  # type: ignore[import-untyped]
    from stable_baselines3.common.vec_env import VecEnv

    from .parallel_rollout_wrapper import RolloutInfoWrapper

    env = RolloutInfoWrapper(env)
    penv = supersuit.vector.MarkovVectorEnv(env, black_death=True)
    venv = supersuit.concat_vec_envs_v1(penv,
                                        num_envs,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")
    return cast(VecEnv, venv)
