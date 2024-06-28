import dataclasses as dc
from typing import Any

import gymnasium as gym
import numpy as np
from navground import core
from navground_learning.core import ModulationActionConfig


@dc.dataclass
class DiscreteModulationActionConfig(ModulationActionConfig):

    param: str = ''
    max_index: int = 1

    def __post_init__(self):
        self.params = {self.param: {'low': 0, 'high': self.max_index, 'discrete': True}}

    @property
    def space(self) -> gym.Space:
        return gym.spaces.Discrete(n=self.max_index + 1)

    def get_params_from_action(self, action: np.ndarray) -> dict[str, int]:
        try:
            value = int(action)  # type: ignore
        except:
            value = int(action[0])
        return {self.param: value}

    def get_action(self, behavior: core.Behavior,
                   time_step: float) -> np.ndarray:
        return np.array(getattr(behavior, self.param), dtype=int)

    def configure(self, behavior: core.Behavior) -> None:
        pass

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs
