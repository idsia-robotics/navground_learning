from __future__ import annotations

from typing import Any

import numpy as np
from imitation.algorithms import dagger
from imitation.data.rollout import make_sample_until
from tqdm.rich import tqdm

from .rollout import generate_trajectories


# Adds
# - supports for experts callable with info
# - tqdm
class SimpleDAggerTrainer(dagger.SimpleDAggerTrainer):

    def train(self,
              total_timesteps: int,
              *,
              rollout_round_min_episodes: int = 3,
              rollout_round_min_timesteps: int = 500,
              bc_train_kwargs: dict[str, Any] | None = None,
              progress_bar: bool = False) -> None:
        total_timestep_count = 0
        round_num = 0

        if progress_bar:
            bar: tqdm[Any] | None = tqdm(total=total_timesteps)
        else:
            bar = None

        while total_timestep_count < total_timesteps:
            collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            sample_until = make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps,
                                  self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

            for traj in trajectories:
                self._logger.record_mean(  # type: ignore[no-untyped-call]
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            self._logger.record(
                "dagger/total_timesteps",
                total_timestep_count)  # type: ignore[no-untyped-call]
            self._logger.record("dagger/round_num",
                                round_num)  # type: ignore[no-untyped-call]
            self._logger.record(
                "dagger/round_episode_count",
                round_episode_count)  # type: ignore[no-untyped-call]
            self._logger.record(
                "dagger/round_timestep_count",
                round_timestep_count)  # type: ignore[no-untyped-call]

            # `logger.dump` is called inside BC.train within the following fn call:
            self.extend_and_update(bc_train_kwargs)
            round_num += 1
            if bar:
                bar.n = min(total_timesteps, total_timestep_count)
                bar.refresh()
