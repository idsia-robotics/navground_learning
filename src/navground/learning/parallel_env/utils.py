from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, cast

from .env import BaseParallelEnv

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def make_vec_from_penv(
    env: BaseParallelEnv,
    num_envs: int = 1,
    processes: int = 1,
    black_death: bool = False,
    seed: int = 0,
    monitor: bool = False,
    monitor_keywords: tuple[str] = ("is_success", )
) -> VecEnv:
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
    :param      seed:       The seed
    :param      black_death: Whether to allow dynamic number of agents
    :param      monitor: Whether to wrap the vector env in a ``VecMonitor``
    :param      monitor_keywords: The keywords passed to ``VecMonitor``

    :returns:   The vector environment with ``number x |agents|``
                single agent environments.
    """
    import supersuit  # type: ignore[import-untyped]
    from stable_baselines3.common.vec_env import VecMonitor

    # penv = supersuit.pettingzoo_env_to_vec_env_v1(env)
    menv = supersuit.vector.MarkovVectorEnv(env, black_death=black_death)
    venv = supersuit.concat_vec_envs_v1(menv,
                                        num_envs,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")

    def get_attr(self: supersuit.vector.MarkovVectorEnv,
                 attr_name: str,
                 indices: None | int | Iterable[int] = None) -> Any:
        if indices is None:
            indices = range(num_envs * menv.num_envs)
        if isinstance(indices, int):
            indices = [indices]
        rs = []
        for i in indices:
            penv = self.vec_envs[i // menv.num_envs].par_env
            rs.append(getattr(penv, attr_name))
        if attr_name == 'render_mode':
            # This is an attribute of vector environments themselves
            return rs[0]
        return rs

    def set_attr(self: supersuit.vector.MarkovVectorEnv,
                 attr_name: str,
                 value: Any,
                 indices: None | int | Iterable[int] = None) -> None:
        if indices is None:
            indices = range(num_envs * menv.num_envs)
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            penv = self.vec_envs[i // menv.num_envs].par_env
            setattr(penv, attr_name, value)

    cenv = venv.unwrapped
    cenv.get_attr = types.MethodType(get_attr, cenv)
    cenv.set_attr = types.MethodType(set_attr, cenv)

    for i, env in enumerate(venv.venv.vec_envs):
        env.reset(seed + i)
    if monitor:
        venv = VecMonitor(venv, info_keywords=monitor_keywords)
    return cast("VecEnv", venv)
