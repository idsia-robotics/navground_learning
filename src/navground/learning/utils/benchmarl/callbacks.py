from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from benchmarl.experiment.callback import Callback  # type: ignore

if TYPE_CHECKING:
    from tensordict import TensorDictBase  # type: ignore


class ExportPolicyCallback(Callback):
    """
    Export the policy. The best model gets exported as ``"best_policy.onnx"``.
    The others as ``"policy_<iterations>.onnx"``.

    :param export_all: Whether to export all (vs just the best) policies.
    """

    def __init__(self, export_all: bool = False):
        super().__init__()
        self.export_all = export_all

    def on_setup(self):
        self.best_reward = -np.inf

    def on_evaluation_end(self, rollouts: list[TensorDictBase]):
        reward = 0
        for group in self.experiment.group_map:
            reward += np.sum([
                np.sum(rollout['next', group, 'reward'].numpy(), axis=0)
                for rollout in rollouts
            ])
        if reward > self.best_reward:
            for group in self.experiment.group_map:
                self.experiment.export_policy(name="best_policy", group=group)
            self.best_reward = reward
        if self.export_all:
            for group in self.experiment.group_map:
                self.experiment.export_policy(
                    name=f"policy_{self.experiment.n_iters_performed}",
                    group=group)


class AlternateActorCallback(Callback):

    def __init__(self,
                 loss: str,
                 learning_rates: list[tuple[int, float]],
                 exclusive: bool = True,
                 group: str = 'agent'):
        super().__init__()
        self._next_actor_index = 0
        self._next_actor_iter = 0
        self._exclusive = exclusive
        self._learning_rates = learning_rates
        self._loss = loss
        self._group = group

    def on_setup(self) -> None:
        # print(self.experiment.algorithm.get_parameters('agent').keys())
        self.all_params = self.experiment.algorithm.get_parameters(self._group)[
            self._loss]

    # TODO(Jerome): should use `on_train_end` instead
    def on_evaluation_end(self, rollouts: list[TensorDictBase]) -> None:
        if self.experiment.n_iters_performed >= self._next_actor_iter:
            iterations, lr = self._learning_rates[self._next_actor_index]
            self._next_actor_iter = self.experiment.n_iters_performed + iterations
            if self._exclusive:
                i = self._next_actor_index
                params = [{
                    'params': self.all_params[6 * i:6 * (i + 1)],
                    'lr': lr,
                    'weight_decay': 0
                }]
                self.experiment.optimizers[self._group][
                    self._loss] = torch.optim.Adam(params)
            else:
                raise NotImplementedError
            self._next_actor_index = (self._next_actor_index + 1) % len(
                self._learning_rates)
