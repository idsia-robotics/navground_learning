from __future__ import annotations

from dataclasses import dataclass

import torch
from benchmarl.models.mlp import Mlp, MlpConfig  # type: ignore
from torch import nn
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensordict import TensorDictBase  # type: ignore


class FlattenMlp(Mlp):  # type: ignore[misc]

    def __init__(self, **kwargs: Any):
        self.flatten = nn.Flatten()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = torch.cat(
            [self.flatten(tensordict.get(in_key)) for in_key in self.in_keys],
            dim=-1)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            res = self.mlp.forward(input)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(input) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class FlattenMlpConfig(MlpConfig):  # type: ignore[misc]

    @staticmethod
    def associated_class() -> type:
        return FlattenMlp
