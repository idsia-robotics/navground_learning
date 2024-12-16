from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy

if TYPE_CHECKING:
    from stable_baselines3.common.type_aliases import PyTorchObs


class RandomPolicy(BasePolicy):
    """
    This class describes a onnx-able policy that returns random actions.
    """

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        if isinstance(observation, dict):
            v = sum(v.sum() for v in observation.values())
            number = next(iter(observation.values())).shape[0]
        else:
            v = observation.sum()
            number = observation.shape[0]
        assert isinstance(self.action_space, gym.spaces.Box)
        high = th.from_numpy(self.action_space.high)
        low = th.from_numpy(self.action_space.low)
        y = th.stack([
            th.rand(self.action_space.shape, device=self.device) *
            (high - low) + low for _ in range(number)
        ])
        y += v * 0
        return y

    def forward(self, x: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(x, deterministic=deterministic)


class SimpleRandomPolicy(BasePolicy):

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        if isinstance(observation, dict):
            number = next(iter(observation.values())).shape[0]
        else:
            number = observation.shape[0]
        actions = np.stack([self.action_space.sample() for _ in range(number)],
                           axis=0)
        return th.as_tensor(actions, device=self.device)

    def forward(self, x: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(x, deterministic=deterministic)
