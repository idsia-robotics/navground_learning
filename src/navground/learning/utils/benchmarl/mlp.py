from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from collections.abc import Callable

import torch

if TYPE_CHECKING:
    from benchmarl.models.mlp import Mlp  # type: ignore

from ...types import PyTorchObs


class MlpPolicy(torch.nn.Module):

    def __init__(self, model: Mlp):
        super().__init__()
        mlp = model.mlp
        self.mlp: Callable[[torch.Tensor], torch.Tensor] = copy.deepcopy(mlp._empty_net)
        mlp.params.to_module(self.mlp)
        self.in_keys = [key[-1] for key in model.in_keys]

    def forward(self, observation: PyTorchObs) -> torch.Tensor:
        if isinstance(observation, dict):
            x: torch.Tensor = torch.cat(
                [observation[key] for key in self.in_keys], dim=-1)
        else:
            x = observation
        return self.mlp(x)
