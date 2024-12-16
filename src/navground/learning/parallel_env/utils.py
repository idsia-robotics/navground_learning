from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .env import BaseParallelEnv

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def make_vec_from_penv(env: BaseParallelEnv,
                       num_envs: int = 1,
                       processes: int = 1) -> VecEnv:
    """
    Converts a :py:class:`PettingZoo  Parallel environment
    <pettingzoo.utils.env.ParallelEnv>`
    to a :py:class:`StableBaseline3 vectorized environment
    <stable_baselines3.common.vec_env.VecEnv>`

    The multiple agents of the single PettingZoo
    (action/observation spaces needs to be shared)
    are stuck as multiple environments of a single agent using
    :py:func:`supersuit.pettingzoo_env_to_vec_env_v1`, and then concatenated
    using :py:func:`supersuit.concat_vec_envs_v1`.

    :param      env:        The environment
    :param      num_envs:   The number of parallel envs to concatenate
    :param      processes:  The number of processes

    :returns:   The vector environment with ``number x |agents|``
                single agent environments.
    """
    import supersuit  # type: ignore[import-untyped]
    from stable_baselines3.common.vec_env import VecEnv

    penv = supersuit.pettingzoo_env_to_vec_env_v1(env)
    penv = supersuit.concat_vec_envs_v1(penv,
                                        num_envs,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")
    return cast(VecEnv, penv)

