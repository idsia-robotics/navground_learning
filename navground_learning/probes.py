from typing import Mapping

from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.types import maybe_wrap_in_dictobs
from navground import core, sim

from .env import GymAgent, GymAgentConfig


class GymProbe(sim.Probe):

    def __init__(self, sensor: sim.Sensor, config: Mapping[int,
                                                           GymAgentConfig]):
        super().__init__()
        self._config = config
        self._sensor = sensor
        self._states = {k: core.SensingState() for k in config}
        for s in self._states.values():
            sensor.prepare(s)

    def prepare(self, run: sim.ExperimentalRun) -> None:
        self._agents = {k: run.world.agents[k] for k in self._config}
        self._gym_agents = {
            k: GymAgent(c, run.world.agents[k].behavior, self._states[k])
            for k, c in self._config.items()
        }
        self._tas = {k: TrajectoryAccumulator() for k in self._config}
        self._first = True
        if self._first:
            self._add_obs(run.world)
            self._first = False

    def update(self, run: sim.ExperimentalRun) -> None:
        for k in self._states:
            self._sensor.update(self._agents[k], run.world, self._states[k])
            obs = self._gym_agents[k].update_observations()
            acts = self._gym_agents[k].get_action(run.time_step)
            self._tas[k].add_step({
                'obs': maybe_wrap_in_dictobs(obs),
                'acts': acts,
                'infos': {},
                'rews': 0.0
            })

    def finalize(self, run: sim.ExperimentalRun) -> None:
        for agent_index, ta in self._tas.items():
            ts = ta.finish_trajectory(None, False)
            for key, data in ts.obs.items():
                run.add_record(f"observations/{agent_index}/{key}", data)
            run.add_record(f"actions/{agent_index}", ts.acts)
            run.add_record(f"rewards/{agent_index}", ts.rews)

    def _add_obs(self, world: sim.World) -> None:
        world._prepare()
        for k in self._states:
            self._sensor.update(self._agents[k], world, self._states[k])
            obs = self._gym_agents[k].update_observations()
            self._tas[k].add_step({'obs': maybe_wrap_in_dictobs(obs)})
