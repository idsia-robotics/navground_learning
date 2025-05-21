from __future__ import annotations

from collections.abc import Sequence

import torch as th

from ..types import PyTorchPolicy


class OnnxablePolicy(th.nn.Module):

    def __init__(self, policy: PyTorchPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.policy(observation, deterministic=True)


class OnnxablePolicyWithMultiInput(th.nn.Module):

    def __init__(self, policy: PyTorchPolicy, keys: Sequence[str]):
        super().__init__()
        self.policy = policy
        self.keys = keys

    def forward(self, *observations: th.Tensor) -> th.Tensor:
        obs = dict(zip(self.keys, list(observations), strict=True))
        return self.policy(obs, deterministic=True)
