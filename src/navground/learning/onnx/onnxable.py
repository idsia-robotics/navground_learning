from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch as th

if TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy


class OnnxablePolicy(th.nn.Module):

    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.policy(observation, deterministic=True))


class OnnxablePolicyWithMultiInput(th.nn.Module):

    def __init__(self, policy: th.nn.Module, keys: Sequence[str]):
        super().__init__()
        self.policy = policy
        self.keys = keys

    def forward(self, *observations: th.Tensor) -> th.Tensor:
        obs = dict(zip(self.keys, list(observations), strict=True))
        return cast(th.Tensor, self.policy(obs, deterministic=True))
