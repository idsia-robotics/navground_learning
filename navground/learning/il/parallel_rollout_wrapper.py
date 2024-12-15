from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from imitation.data import types
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from ..internal.base_env import ResetReturn, StepReturn
from ..types import Action, Observation

if TYPE_CHECKING:
    from pettingzoo.utils.env import ParallelEnv

ObsType = types.DictObs | np.typing.NDArray[Any]

EnvKwargs = Mapping[str, Any]


class RolloutInfoWrapper(BaseParallelWrapper[int, Observation, Action]):

    def __init__(self, env: ParallelEnv[int, Observation, Action]):
        super().__init__(env)
        self._obs: dict[int, list[types.DictObs]
                        | list[np.typing.NDArray[Any]]] = {}
        self._rews: dict[int, list[float]] = {}

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:
        obss, infos = self.env.reset(seed=seed, options=options)
        self._obs = {
            i: [types.maybe_wrap_in_dictobs(obs)]  # type: ignore[misc]
            for i, obs in obss.items()
        }
        self._rews = {i: [] for i in obss}
        return obss, infos

    def step(self, actions: dict[int, Action]) -> StepReturn:
        obss, rews, terms, trucs, infos = self.env.step(actions)
        for i, obs in obss.items():
            self._obs[i].append(
                types.maybe_wrap_in_dictobs(obs))  # type: ignore[arg-type]
        for i, rew in rews.items():
            self._rews[i].append(rew)

        for i, term in terms.items():
            done = term or trucs[i]
            if done:
                info = infos[i]
                assert "rollout" not in info
                info["rollout"] = {
                    "obs": types.stack_maybe_dictobs(
                        self._obs[i]),  # type: ignore[misc]
                    "rews": np.stack(self._rews[i]),
                }
        return obss, rews, terms, trucs, infos
