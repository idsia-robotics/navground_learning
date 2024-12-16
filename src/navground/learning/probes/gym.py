from __future__ import annotations

import functools
from collections.abc import Callable, Collection
from typing import cast


from navground import sim

from ..config import GroupConfig
from ..env import BaseEnv
from ..internal.base_env import NavgroundBaseEnv
from ..internal.group import create_agents_in_groups
from ..parallel_env import BaseParallelEnv


# TODO(Jerome): check that it works with flat observations
class GymProbe(sim.Probe):
    """
    A probe to record observation, rewards and actions, like
    during a rollout. Internally uses a
    :py:class:`imitation.data.rollout.TrajectoryAccumulator`
    to store the data, which it writes to datasets
    only at the end of the run:

    - observations/<agent_index>/<key>
    - actions/<agent_index>
    - rewards/<agent_index>

    :param      groups:  The configuration of the groups

    """

    def __init__(self, groups: Collection[GroupConfig]):
        super().__init__()
        self._groups = groups

    def prepare(self, run: sim.ExperimentalRun) -> None:
        from imitation.data.rollout import TrajectoryAccumulator

        self._agents = create_agents_in_groups(run.world, self._groups)
        self._tas = {
            k: TrajectoryAccumulator()  # type: ignore[no-untyped-call]
            for k in self._agents
        }
        self._first = True
        if self._first:
            self._add_obs(run.world)
            self._first = False

    def update(self, run: sim.ExperimentalRun) -> None:
        from imitation.data.types import maybe_wrap_in_dictobs

        for i, agent in self._agents.items():
            agent.update_state(run.world)
            if agent.gym:
                obs = agent.gym.update_observation()
                acts = agent.gym.get_action(run.time_step)
                self._tas[i].add_step({
                    'obs': maybe_wrap_in_dictobs(obs),
                    'acts': acts,
                    'infos': {},
                    'rews': 0.0  # type: ignore
                })

    def finalize(self, run: sim.ExperimentalRun) -> None:
        from imitation.data.types import DictObs

        for agent_index, ta in self._tas.items():
            ts = ta.finish_trajectory(None, False)
            for key, data in cast(
                    DictObs, ts.obs).items():  # type: ignore[no-untyped-call]
                run.add_record(f"observations/{agent_index}/{key}", data)
            run.add_record(f"actions/{agent_index}", ts.acts)
            run.add_record(f"rewards/{agent_index}", ts.rews)

    def _add_obs(self, world: sim.World) -> None:
        from imitation.data.types import maybe_wrap_in_dictobs

        world._prepare()
        for i, agent in self._agents.items():
            agent.update_state(world)
            if agent.gym:
                obs = agent.gym.update_observation()
                self._tas[i].add_step({'obs': maybe_wrap_in_dictobs(obs)})

    @classmethod
    def with_env(cls, env: BaseEnv | BaseParallelEnv) -> Callable[[], GymProbe]:
        """
        Creates a probe factory to record all actions and observations
        in an environment

        :param      env:  The environment

        :returns:   A callable that can be added to runs or experiments,
            using :py:meth:`navground.sim.ExperimentalRun.add_record_probe` or
            :py:meth:`navground.sim.Experiment.add_record_probe`
        """
        assert isinstance(env.unwrapped, NavgroundBaseEnv)
        return functools.partial(cls, groups=env.unwrapped.groups_config)
