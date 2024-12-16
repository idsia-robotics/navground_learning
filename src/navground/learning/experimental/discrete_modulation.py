from __future__ import annotations

import dataclasses as dc
from typing import Any

import gymnasium as gym
import numpy as np
from navground import core

from ..config import ModulationActionConfig
from ..types import Action


@dc.dataclass
class DiscreteModulationActionConfig(ModulationActionConfig):

    param: str = ''
    max_index: int = 1

    def __post_init__(self) -> None:
        self.params = {self.param: {'low': 0, 'high': self.max_index, 'discrete': True}}

    @property
    def space(self) -> gym.Space[Any]:
        return gym.spaces.Discrete(n=self.max_index + 1)

    def get_params_from_action(self, action: Action) -> dict[str, int]:
        try:
            value = int(action)
        except Exception:
            value = int(action[0])
        return {self.param: value}

    def get_action(self, behavior: core.Behavior,
                   time_step: float) -> Action:
        return np.array(getattr(behavior, self.param), dtype=int)

    def configure(self, behavior: core.Behavior) -> None:
        pass

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs
