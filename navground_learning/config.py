import dataclasses as dc
from collections.abc import Sequence
from operator import itemgetter
from typing import Any, TypeVar, cast

import yaml
from navground import core, sim

from .core import ActionConfig, Agent, GymAgent, ObservationConfig
from .reward import Reward

Indices = list[int] | slice | None

T = TypeVar("T")


def get_elements_at(indices: Indices, xs: Sequence[T]) -> Sequence[T]:
    if indices is None:
        return xs
    if isinstance(indices, list):
        if len(indices) == 1:
            return [xs[indices[0]]]
        if len(indices) > 1:
            return itemgetter(*indices)(xs)
        return []
    return xs[indices]


def to_list(indices: Indices, xs: Sequence[T]) -> list[int]:
    if indices is None:
        return list(range(len(xs)))
    if isinstance(indices, list):
        return [i for i in indices if i >= 0 and i < len(xs)]
    return to_list(list(range(indices.start, indices.stop, indices.step or 1)),
                   xs)


def lowest(indices: Indices) -> int | None:
    if indices is None:
        return 0
    if isinstance(indices, list):
        for i in indices:
            if i >= 0:
                return i
        return None
    i = indices.start
    while i < indices.stop:
        if i >= 0:
            return i
        i += indices.step
    return None


def make_sensor(
        value: sim.Sensor | str | dict[str, Any] | None) -> sim.Sensor | None:
    if isinstance(value, dict):
        value = yaml.dump(value)
    if isinstance(value, str):
        se = sim.load_state_estimation(value)
        if isinstance(se, sim.Sensor):
            value = se
        else:
            value = None
            print(f"Invalid sensor {se}")
    return cast(sim.Sensor | None, value)


def get_sensor_as_dict(sensor: sim.Sensor | None | str | dict) -> dict[str, Any]:
    if sensor is None:
        return {}
    if isinstance(sensor, dict):
        return sensor
    if isinstance(sensor, str):
        return yaml.safe_load(sensor)
    return yaml.safe_load(sim.dump(sensor))


@dc.dataclass
class GroupConfig:
    """
    :param indices: The world indices of the agents.

    :param sensor: A sensor to produce observations for the agents.
                   If a :py:class:`str`, it will be interpreted as the YAML
                   representation of a sensor.
                   If a :py:class:`dict`, it will be dumped to YAML and
                   then treated as a :py:class:`str`.
                   If None, it will use the agents' own state estimation, if a sensor.

    :param action: The configuration of the action and observation space to use.

    :param observation: The configuration of the observation space to use.

    :param reward: The reward function to use.
    """
    action: ActionConfig | None = None
    indices: Indices = None
    sensor: sim.Sensor | str | dict[str, Any] | None = None
    reward: Reward | None = None
    observation: ObservationConfig | None = None

    @property
    def asdict(self) -> dict[str, Any]:
        rs: dict[str, Any] = {}
        if self.observation:
            rs['observation'] = dc.asdict(self.observation)
        if self.action:
            rs['action'] = dc.asdict(self.action)
        if self.reward:
            rs['reward'] = self.reward.asdict
        if self.indices is None or isinstance(self.indices, list):
            rs['indices'] = self.indices
        else:
            rs['indices'] = {
                'start': self.indices.start,
                'stop': self.indices.stop,
                'step': self.indices.step}
        rs['sensor'] = get_sensor_as_dict(self.sensor)
        return rs

    def get_sensor(self) -> sim.Sensor | None:
        return make_sensor(self.sensor)


@dc.dataclass
class WorldConfig:
    groups: list[GroupConfig] = dc.field(default_factory=list)
    policies: list[tuple[Indices, Any]] = dc.field(default_factory=list)
    reward: Reward | None = None

    @property
    def asdict(self) -> dict[str, Any]:
        rs: dict[str, Any] = {'groups': [g.asdict for g in self.groups]}
        if self.reward:
            rs['reward'] = self.reward.asdict
        return rs

    def get_first_reward(self) -> Reward | None:
        for group in self.groups:
            if group.reward:
                return group.reward
        return None

    def init_agents(
            self,
            world: sim.World,
            max_number_of_agents: int | None = None) -> dict[int, Agent]:
        agents = {}
        for group in self.groups:
            sensor = group.get_sensor()
            indices = to_list(group.indices, world.agents)
            representative_behavior = world.agents[
                indices[0]].behavior if indices else None
            if max_number_of_agents is not None and max_number_of_agents > len(
                    world.agents):
                indices = to_list(group.indices,
                                  list(range(max_number_of_agents)))
            for i in indices:
                if sensor:
                    state = core.SensingState()
                    sensor.prepare(state)
                else:
                    state = None
                if i >= 0 and i < len(world.agents):
                    ng_agent = world.agents[i]
                    behavior: core.Behavior | None = ng_agent.behavior
                else:
                    ng_agent = None
                    behavior = representative_behavior
                if group.observation and group.action:
                    gym_agent = GymAgent(observation=group.observation,
                                         action=group.action,
                                         behavior=behavior,
                                         state=state)
                else:
                    gym_agent = None
                agents[i] = Agent(state=state,
                                  gym=gym_agent,
                                  sensor=sensor,
                                  reward=group.reward,
                                  navground=ng_agent)
        for p_indices, policy in self.policies:
            for i in to_list(p_indices, world.agents):
                if i in agents and agents[i].gym:
                    agents[i].policy = policy
        if self.reward:
            for i, ng_agent in enumerate(world.agents):
                if i not in agents:
                    agents[i] = Agent(reward=self.reward)
        return agents
