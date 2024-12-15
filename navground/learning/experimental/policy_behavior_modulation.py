from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from functools import partial
from typing import Any, cast

from navground import core, sim

from ..config import (DefaultObservationConfig, GroupConfig,
                      ModulationActionConfig, ObservationConfig)
from ..internal.base_env import NavgroundBaseEnv
from ..internal.group import GymAgent, create_agents_in_groups
from ..probes import RewardProbe


def update_state(sensor: sim.Sensor | None, world: sim.World,
                 agent: sim.Agent) -> Callable[[core.SensingState], None]:

    def f(state: core.SensingState) -> None:
        if sensor:
            sensor.update(agent, world, state)

    return f


class PolicyModulation(core.BehaviorModulation):

    def __init__(
            self,
            action_config: ModulationActionConfig = ModulationActionConfig(),
            observation_config: ObservationConfig = DefaultObservationConfig(),
            policy: Any = None,
            observation_fn: Callable[[core.SensingState], None] | None = None,
            deterministic: bool = True):
        super().__init__()
        self.action_config = action_config
        self.observation_config = observation_config
        self.gym_agent: GymAgent | None = None
        self.policy = policy
        self.observation_fn = observation_fn
        self.state = core.SensingState()
        self.deterministic = deterministic
        self._old_params: dict[str, Any] = {}

    def pre(self, behavior: core.Behavior, time_step: float) -> None:
        if self.observation_fn:
            self.observation_fn(self.state)
        if not self.gym_agent:
            self.gym_agent = GymAgent(action=self.action_config,
                                      observation=self.observation_config,
                                      state=self.state,
                                      behavior=behavior)
        obs = self.gym_agent.update_observation()
        act, _ = self.policy.predict(obs, deterministic=self.deterministic)
        ac = cast(ModulationActionConfig, self.gym_agent.action_config)
        params = ac.get_params_from_action(act)
        for k, v in params.items():
            self._old_params[k] = getattr(behavior, k)
            setattr(behavior, k, v)
        behavior.modulated_params = params  # type: ignore
        behavior.modulation_input = obs  # type: ignore

    def post(self, behavior: core.Behavior, time_step: float,
             cmd: core.Twist2) -> core.Twist2:
        for k, v in self._old_params.items():
            setattr(behavior, k, v)
        return cmd


def add_modulation(groups: Collection[GroupConfig],
                   policy: Any) -> Callable[[sim.World, int | None], None]:

    def f(world: sim.World, seed: int | None = None) -> None:
        for group in groups:
            for agent in group.indices.sub_sequence(world.agents):
                obs = update_state(group.get_sensor(), world, agent)
                if group.action and isinstance(
                        group.action,
                        ModulationActionConfig) and group.observation:
                    agent.behavior.add_modulation(  # type: ignore
                        PolicyModulation(group.action, group.observation,
                                         policy, obs))

    return f


# Not safe: assumes all agents have the same modulation
class ModProbe(sim.RecordProbe):

    dtype = float

    def __init__(self, ds: sim.Dataset, groups: Collection[GroupConfig]):
        super().__init__(ds)
        self._groups = groups
        self.sizes: dict[int, int] = {}

    def prepare(self, run: sim.ExperimentalRun) -> None:
        agents = create_agents_in_groups(run.world, self._groups)
        for index, agent in agents.items():
            if agent.navground and agent.navground.behavior:
                self.sizes[index] = sum([
                    len(mod.action_config.params)
                    for mod in agent.navground.behavior.modulations
                    if isinstance(mod, PolicyModulation)
                ])
        if not len(set(self.sizes.values())) < 2:
            raise ValueError(f"Unnormal param sizes {self.sizes}")
        super().prepare(run)

    def update(self, run: sim.ExperimentalRun) -> None:

        for agent in run.world.agents:
            if agent.behavior and hasattr(agent.behavior, "modulated_params"):
                params = agent.behavior.modulated_params
                for v in params.values():
                    self.data.push(v)

    def get_shape(self, world: sim.World) -> list[int]:
        sizes = list(self.sizes.values())
        if sizes:
            size = sizes[0]
        else:
            size = 0
        return [len(self.sizes), size]


def make_experiment_with_env(
    env: NavgroundBaseEnv,
    policy: Any = None,
    record_modulations: bool = False,
    terminate_when_all_idle_or_stuck: bool = False
    # record_obervations: bool = False
) -> sim.Experiment:
    exp = sim.Experiment()
    if env._scenario:
        exp.scenario = copy.copy(env._scenario)
    else:
        raise ValueError("No scenario")
    if policy:
        exp.scenario.add_init(add_modulation(env.groups_config, policy))
        if record_modulations:
            exp.add_record_probe("modulation",
                                 partial(ModProbe, groups=env.groups_config))
        # if record_obervations:
        #     exp.add_record_probe("observation",
        #                          partial(ObsProbe, config=env.config))
    exp.add_record_probe("reward",
                         partial(RewardProbe, groups=env.groups_config))
    exp.terminate_when_all_idle_or_stuck = terminate_when_all_idle_or_stuck
    return exp
