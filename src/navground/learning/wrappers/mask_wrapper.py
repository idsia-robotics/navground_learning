from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

if TYPE_CHECKING:
    from pettingzoo.utils.env import ParallelEnv

from ..internal.base_env import ResetReturn, StepReturn
from ..types import Action, Array, Observation


def mask_array_or_dict(
    value: dict[str, Array] | Array,
    indices: Iterable[int] = tuple(),
    keys: Iterable[str] = tuple()
) -> Observation:
    if isinstance(value, Mapping):
        keys = set(keys)
        return {k: v for k, v in value.items() if k not in keys}
    indices = list(indices)
    if indices:
        return np.delete(value, indices, axis=-1)
    return value


def unmask_array(values: Array,
                 indices: Iterable[int],
                 value: float = 0) -> Array:
    for i in indices:
        values = np.insert(values, i, value, axis=-1)
    return values


def mask_space(
    value: gym.spaces.Box | gym.spaces.Dict,
    indices: Iterable[int] = tuple(),
    keys: Iterable[str] = tuple()
) -> gym.spaces.Box | gym.spaces.Dict:
    if isinstance(value, Mapping):
        keys = set(keys)
        return gym.spaces.Dict({
            k: v
            for k, v in value.items() if k not in keys
        })
    indices = list(indices)
    return gym.spaces.Box(np.delete(value.low, indices),
                          np.delete(value.high, indices))


class MaskWrapper(BaseParallelWrapper):
    """
    This wrapper masks some of the observations and/or actions

    :param env: The wrapped environment
    :param observation_indices: Which observation indices to keep (if array)
    :param observation_keys: Which observation keys to keep (if dict)
    :param action_indices: Which action indices to keep (if array)
    """

    def __init__(
        self,
        env: ParallelEnv,
        observation_indices: Iterable[int] = tuple(),
        observation_keys: Iterable[str] = tuple(),
        action_indices: Iterable[int] = tuple()
    ) -> None:
        super().__init__(env)
        self._action_indices = action_indices
        self._observation_indices = observation_indices
        self._observation_keys = observation_keys

    def observation_space(self, agent):
        return mask_space(self.env.observation_space(agent),
                          indices=self._observation_indices,
                          keys=self._observation_keys)

    def action_space(self, agent):
        return mask_space(self.env.action_space(agent),
                          indices=self._action_indices)

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:
        obs, infos = self.env.reset(seed=seed, options=options)
        obs = {
            agent:
            mask_array_or_dict(o,
                               indices=self._observation_indices,
                               keys=self._observation_keys)
            for agent, o in obs.items()
        }
        return obs, infos

    def step(self, action: dict[int, Action]) -> StepReturn:
        action = {
            agent: unmask_array(act, indices=self._action_indices)
            for agent, act in action.items()
        }
        obs, reward, terminated, truncated, infos = self.env.step(action)
        obs = {
            agent:
            mask_array_or_dict(o,
                               indices=self._observation_indices,
                               keys=self._observation_keys)
            for agent, o in obs.items()
        }
        return obs, reward, terminated, truncated, infos
