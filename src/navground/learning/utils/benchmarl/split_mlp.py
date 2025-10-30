from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tensordict import TensorDictBase  # type: ignore
    from torch import nn
    from benchmarl.experiment import Experiment  # type: ignore
    from torchrl.data.tensor_specs import TensorSpec  # type: ignore

import copy
from collections import ChainMap
from collections.abc import Collection, Sequence
from dataclasses import MISSING, dataclass, field

import torch
from torchrl.modules import MultiAgentMLP  # type: ignore

from benchmarl.models.common import Model, ModelConfig  # type: ignore

from ...types import PyTorchObs

InputSpec = slice | Collection[str] | None
MlpSpec = tuple[int, InputSpec, dict[str, Any]]


def compute_input_size(input_spec: TensorSpec, i_spec: InputSpec,
                       keys: Sequence[str]) -> int:
    if i_spec is None:
        return sum([spec.shape[-1] for spec in input_spec])
    if isinstance(i_spec, slice):
        return sum(
            [len(list(range(spec.shape[-1]))[i_spec]) for spec in input_spec])
    return sum([
        spec.shape[-1] for spec, key in zip(input_spec, keys, strict=True)
        if key[-1] in i_spec
    ])

    # self.input_features =


def compute_input(tensordict: TensorDictBase, i_spec: InputSpec,
                  keys: Sequence[str]) -> torch.Tensor:
    # print('compute_input', i_spec, keys)
    if i_spec is None:
        return torch.cat([tensordict.get(in_key) for in_key in keys], dim=-1)
    if isinstance(i_spec, slice):
        return torch.cat([tensordict.get(in_key) for in_key in keys],
                         dim=-1)[..., i_spec]
    return torch.cat(
        [tensordict.get(in_key) for in_key in keys if in_key[-1] in i_spec],
        dim=-1)


class SplitMlp(Model):  # type: ignore[misc]

    def __init__(
        self,
        mlps_specs: Sequence[MlpSpec] = [],
        **kwargs: Any,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        assert self.input_has_agent_dim is True

        out_dim = self.output_leaf_spec.shape[-1]
        mlp_out_dim = sum(spec[0] for spec in mlps_specs)
        k = out_dim // mlp_out_dim

        self.mlps_specs = mlps_specs
        input_sizes = [
            compute_input_size(self.input_spec.values(True, True), input_spec,
                               self.in_keys) for _, input_spec, _ in mlps_specs
        ]
        self.mlps = [
            MultiAgentMLP(n_agent_inputs=input_size,
                          n_agent_outputs=k * dims,
                          n_agents=self.n_agents,
                          centralised=self.centralised,
                          share_params=self.share_params,
                          device=self.device,
                          **ChainMap(i_kwargs, kwargs))
            for (dims, _, i_kwargs
                 ), input_size in zip(mlps_specs, input_sizes, strict=True)
        ]
        for i, mlp in enumerate(self.mlps):
            self.add_module(str(i), mlp)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        outs = []
        for mlp, (_, spec, _) in zip(self.mlps, self.mlps_specs, strict=True):
            input = compute_input(tensordict, spec, self.in_keys)
            out = mlp.forward(input)
            shape = out.shape
            outs.append(out)
        res = torch.stack(outs, dim=-1).reshape((*shape[:-1], -1))
        if not self.output_has_agent_dim:
            res = res[..., 0, :]
        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class SplitMlpConfig(ModelConfig):  # type: ignore[misc]
    """
    A model with a independent MLP sub-models for different dimensions
    of the action space, as specified by ``mlps_specs``, a list of
    tuples ``(action_size, input_spec, net_arch)``
    each configure one of the MLPs that computes ``action_size`` actions
    using observations specified by``input_spec`` with an architecture ``net_arch``,
    similar to how :py:class:navground.learning.policies.SplitSACPolicy` is
    configured.

    The list should be ordered and actions sizes
    should sum up to the total size of the action space.


    For example:

    >>> env.observation_space('agent_0')
    Dict('a': Box(0.0, 1.0, (1,), float32), 'b': Box(0.0, 1.0, (1,), float32))
    >>> env.action_space('agent_0')
    Box(-1.0, 1.0, (2,), float32)
    >>> mlps_specs = [(1, None, {'num_cells': [64, 64]}), (1, ['a'],  {'num_cells': [16, 16]})]
    >>> model_config = SplitMlpConfig(mlps_specs=mlps_specs, ...)

    configures a model that computes two actions:

    - the first using a 64 + 64 MLP from ``a`` and ``b``
    - the second using a 16 + 16 MLP solely from ``a``

    """
    num_cells: Sequence[int] = MISSING  # type: ignore[assignment]
    layer_class: type[nn.Module] = MISSING  # type: ignore[assignment]

    activation_class: type[nn.Module] = MISSING  # type: ignore[assignment]
    activation_kwargs: dict[str, Any] | None = None

    norm_class: type[nn.Module] | None = None
    norm_kwargs: dict[str, Any] | None = None

    mlps_specs: list[MlpSpec] = field(default_factory=list)

    @staticmethod
    def associated_class() -> type:
        return SplitMlp


# https://pytorch.org/docs/main/optim.html#per-parameter-options


def select_submlp(experiment: Experiment,
                  index: int,
                  lr: float = 1e-4,
                  eps: float = 1e-6) -> None:
    split_mlp = experiment.policy[0][0][0]
    mlp = split_mlp.mlps[index]
    experiment.optimizers['agent']['loss_objective'] = torch.optim.Adam(
        mlp.parameters(), eps=eps, lr=lr)


def setup_optimizer(experiment: Experiment,
                    lrs: Sequence[float],
                    eps: float = 1e-6) -> None:
    split_mlp = experiment.policy[0][0][0]
    params = [{
        'params': mlp.parameters(),
        'lr': lr
    } for mlp, lr in zip(split_mlp.mlps, lrs, strict=True)]
    experiment.optimizers['agent']['loss_objective'] = torch.optim.Adam(
        params, eps=eps)


def set_lr(experiment: Experiment, index: int, lr: float) -> None:
    experiment.optimizers['agent']['loss_objective'].param_groups[index][
        'lr'] = lr


def compute_input_2(obs: PyTorchObs, i_spec: InputSpec,
                    keys: Sequence[str]) -> torch.Tensor:
    if isinstance(obs, dict):
        if i_spec is None:
            return torch.cat([obs[in_key] for in_key in keys], dim=-1)
        return torch.cat([
            obs[in_key]
            for in_key in keys if in_key in cast("Collection[str]", i_spec)
        ],
                         dim=-1)
    if i_spec is None:
        return obs
    return obs[..., cast("slice", i_spec)]


class SplitMlpPolicy(torch.nn.Module):

    def __init__(self, model: SplitMlp):
        super().__init__()
        self._specs = [x for _, x, _ in model.mlps_specs]
        self._mlp_nns = []
        self.in_keys = [x[-1] for x in model.in_keys]
        for mlp in model.mlps:
            mlp_nn = copy.deepcopy(mlp._empty_net)
            mlp.params.to_module(mlp_nn)
            self._mlp_nns.append(mlp_nn)
        for i, mlp in enumerate(self._mlp_nns):
            self.add_module(f'module_{i}', mlp)

    def forward(self, observation: PyTorchObs) -> torch.Tensor:
        logits = [
            mlp(compute_input_2(observation, spec, self.in_keys))
            for mlp, spec in zip(self._mlp_nns, self._specs, strict=True)
        ]
        return torch.stack(logits, dim=-1).reshape((*logits[0].shape[:-1], -1))
