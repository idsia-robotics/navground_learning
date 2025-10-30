from __future__ import annotations

from typing import Any
from collections.abc import Callable

import gymnasium as gym
import torch
from torch.distributions import transforms as D
from torch.distributions.categorical import Categorical
from torchrl.modules import ProbabilisticActor  # type: ignore
from torchrl.modules.distributions.continuous import TanhNormal  # type: ignore
from torchrl.modules.tensordict_module import SafeProbabilisticModule  # type: ignore

from benchmarl.models.mlp import Mlp  # type: ignore

from ...types import (Action, Array, EpisodeStart, Info, Observation,
                      PyTorchObs, State)
from .mlp import MlpPolicy
from .split_mlp import SplitMlp, SplitMlpPolicy


def _has_batch_dim(observation: Array, flat_dims: int) -> bool:
    return len(observation.shape) == 2 or observation.shape[0] != flat_dims

class SingleAgentPolicy(torch.nn.Module):
    """
    This class conforms to :py:class:`navground.learning.types.PyTorchPolicy`
    and is construction from a TorchRL policy.
    """

    def __init__(self, observation_space: gym.Space[Any],
                 action_space: gym.spaces.Box, policy: Any):
        """
        Constructs a new instance.

        :param      observation_space:  The observation space
        :param      action_space:       The action space
        :param      policy:             The TorchRL policy
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        # actor = policy.module[0]
        actor = policy
        dist_module = actor[1]
        self._tail: Callable[[torch.Tensor], torch.Tensor]
        if isinstance(dist_module, SafeProbabilisticModule):
            if dist_module.distribution_class in (TanhNormal, ):
                min_act = dist_module.distribution_kwargs['low'][0]
                max_act = dist_module.distribution_kwargs['high'][0]
                t = D.AffineTransform(loc=(max_act + min_act) / 2,
                                      scale=(max_act - min_act) / 2)
                self._tail = lambda x: t(x.tanh())
                # self._tail = lambda x: t(dist_module(x))
            elif dist_module.distribution_class in (Categorical, ):
                self._tail = lambda x: dist_module(x)[0]
            else:
                self._tail = lambda x: dist_module(x)
        else:
            self._tail = lambda x: dist_module(x)[0]
        self._npe = None
        if isinstance(actor, ProbabilisticActor):
            if isinstance(action_space, gym.spaces.Box):
                model = actor.module[0][0]
                self._npe = actor[0][1].module
            else:
                model = actor.module[0]
        else:
            model = actor.module[0]
        self.model = self._copy_model(model)
        if isinstance(observation_space, gym.spaces.Dict):
            self._uses_dict = True
            self._flat_dims = {
                k: gym.spaces.utils.flatdim(v)
                for k, v in observation_space.items()
            }
        else:
            self._uses_dict = False
            self._flat_dim = gym.spaces.utils.flatdim(observation_space)

    def forward(self,
                observation: PyTorchObs,
                deterministic: bool = True) -> torch.Tensor:
        logits = self.model.forward(observation)
        if self._npe:
            loc, _ = self._npe(logits)
        else:
            loc = logits
        return self._tail(loc)

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False,
                info: Info | None = None) -> tuple[Action, State | None]:
        if self._uses_dict:
            if not isinstance(observation, dict):
                raise TypeError("Requires a dict observation")
            are_batches = set(
                _has_batch_dim(v, self._flat_dims[k])
                for k, v in observation.items())
            is_batch = all(are_batches)
            obs: PyTorchObs = {
                k: torch.from_numpy(v.reshape(-1, self._flat_dims[k]))
                for k, v in observation.items()
            }
        else:
            if isinstance(observation, dict):
                raise TypeError("Requires a flat observation")
            is_batch = _has_batch_dim(observation, self._flat_dim)
            obs = torch.from_numpy(observation.reshape(-1, self._flat_dim))
        with torch.no_grad():
            act = self.forward(obs,
                               deterministic=deterministic).detach().numpy()
            if not is_batch:
                act = act.flatten()
            return act, None

    def _copy_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, Mlp):
            return MlpPolicy(model)
        if isinstance(model, SplitMlp):
            return SplitMlpPolicy(model)
        raise TypeError(f"Model {type(model)} not supported")
