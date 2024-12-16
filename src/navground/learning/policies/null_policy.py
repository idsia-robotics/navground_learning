from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy

if TYPE_CHECKING:
    from stable_baselines3.common.type_aliases import PyTorchObs


class NullPolicy(BasePolicy):
    """
    This class describes a dummy policy that always returns zeros.
    """
    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> th.Tensor:
        if isinstance(observation, dict):
            number = next(iter(observation.values())).shape[0]
        else:
            number = observation.shape[0]
        action = np.zeros(
            shape=(number, *self.action_space.shape),  # type: ignore[misc]
            dtype=self.action_space.dtype)
        return th.as_tensor(action)

    def forward(self, x: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(x, deterministic=deterministic)
