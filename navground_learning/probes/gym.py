from typing import cast

from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.types import DictObs, maybe_wrap_in_dictobs
from navground import sim

from ..config import WorldConfig


class GymProbe(sim.Probe):

    def __init__(self, config: WorldConfig):
        super().__init__()
        self._config = config

    def prepare(self, run: sim.ExperimentalRun) -> None:
        self._agents = self._config.init_agents(run.world)
        self._tas = {k: TrajectoryAccumulator() for k in self._agents}
        self._first = True
        if self._first:
            self._add_obs(run.world)
            self._first = False

    def update(self, run: sim.ExperimentalRun) -> None:
        for i, agent in self._agents.items():
            agent.update_state(run.world)
            if agent.gym:
                obs = agent.gym.update_observations()
                acts = agent.gym.get_action(run.time_step)
                self._tas[i].add_step({
                    'obs': maybe_wrap_in_dictobs(obs),
                    'acts': acts,
                    'infos': {},
                    'rews': 0.0  # type: ignore
                })

    def finalize(self, run: sim.ExperimentalRun) -> None:
        for agent_index, ta in self._tas.items():
            ts = ta.finish_trajectory(None, False)
            for key, data in cast(DictObs, ts.obs).items():
                run.add_record(f"observations/{agent_index}/{key}", data)
            run.add_record(f"actions/{agent_index}", ts.acts)
            run.add_record(f"rewards/{agent_index}", ts.rews)

    def _add_obs(self, world: sim.World) -> None:
        world._prepare()
        for i, agent in self._agents.items():
            agent.update_state(world)
            if agent.gym:
                obs = agent.gym.update_observations()
                self._tas[i].add_step({'obs': maybe_wrap_in_dictobs(obs)})
