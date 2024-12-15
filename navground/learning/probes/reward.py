from __future__ import annotations

import functools
from collections.abc import Callable, Collection
from typing import cast

import numpy as np
from navground import sim

from ..config import GroupConfig
from ..internal.group import create_agents_in_groups
from ..types import Reward


class RewardProbe(sim.RecordProbe):
    """
    A probe to record rewards to a single dataset
    of shape ``(#agents, #steps)``

    :param      ds:      The dataset
    :param      groups:  The configuration of the groups
    :param      reward:  The reward function

    """

    dtype = np.float64

    def __init__(self,
                 ds: sim.Dataset,
                 groups: Collection[GroupConfig] = tuple(),
                 reward: Reward | None = None):
        super().__init__(ds)
        self._groups = groups
        self._reward: dict[int, Reward] = {}
        self._default_reward = reward

    def prepare(self, run: sim.ExperimentalRun) -> None:
        agents = create_agents_in_groups(run.world, self._groups)
        for index, agent in agents.items():
            if agent.reward is not None:
                self._reward[index] = agent.reward
        super().prepare(run)

    def update(self, run: sim.ExperimentalRun) -> None:
        for i, agent in enumerate(run.world.agents):
            reward = self._reward.get(i, self._default_reward)
            if reward:
                value = reward(agent=agent,
                               world=run.world,
                               time_step=run.time_step)
                self.data.push(value)

    def get_shape(self, world: sim.World) -> list[int]:
        if self._default_reward:
            return [len(world.agents)]
        return [len(self._reward)]

    @classmethod
    def with_reward(cls,
                    reward: Reward) -> Callable[[sim.Dataset], RewardProbe]:
        """
        Creates a probe factory to record a reward

        :param      reward:  The reward

        :returns:   A callable that can be added to runs or experiments using
            :py:meth:`navground.sim.ExperimentalRun.add_record_probe` or
            :py:meth:`navground.sim.Experiment.add_record_probe`.
        """
        probe = functools.partial(cls, reward=reward)
        return cast(Callable[[sim.Dataset], 'RewardProbe'], probe)
