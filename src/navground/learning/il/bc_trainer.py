from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from imitation.algorithms import bc
from imitation.data import rollout as rollout_without_info

from . import rollout

if TYPE_CHECKING:
    import numpy as np
    from stable_baselines3.common import policies


class RolloutStatsComputer(bc.RolloutStatsComputer):

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        rng: np.random.Generator,
    ) -> Mapping[str, float]:
        if self.venv is not None and self.n_episodes > 0:
            trajs = rollout.generate_trajectories(
                policy,
                self.venv,
                rollout_without_info.make_min_episodes(self.n_episodes),
                rng=rng,
            )
            return rollout_without_info.rollout_stats(trajs)
        else:
            return dict()


# TODO(Jerome): monkey patching for now
# In the future, better to subclass BC.
bc.RolloutStatsComputer = RolloutStatsComputer  # type: ignore[misc]

BC = bc.BC

__all__ = ['BC', 'RolloutStatsComputer']
