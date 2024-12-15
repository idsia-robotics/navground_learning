from __future__ import annotations

import copy
import dataclasses as dc
import numbers
from collections import deque
from collections.abc import Collection, Mapping
from typing import Any, Protocol
import warnings

import gymnasium as gym
import numpy as np
from navground import core, sim

from ..config import GroupConfig
from ..types import Array, Reward


class ActionProtocol(Protocol):

    def configure(self, behavior: core.Behavior) -> None:
        ...

    def is_configured(self, warn: bool = False) -> bool:
        ...

    @property
    def space(self) -> gym.Space[Any]:
        ...

    def get_cmd_from_action(self, action: Array,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        ...

    def get_action(self, behavior: core.Behavior, time_step: float) -> Array:
        ...


class ObservationProtocol(Protocol):

    def configure(self, behavior: core.Behavior | None,
                  sensing_space: gym.spaces.Dict) -> None:
        ...

    def is_configured(self, warn: bool = False) -> bool:
        ...

    def get_observation(
            self, behavior: core.Behavior | None,
            buffers: Mapping[str, core.Buffer]) -> dict[str, Array] | Array:
        ...

    @property
    def space(self) -> gym.Space[Any]:
        ...

    @property
    def history(self) -> int:
        ...


def make_simple_space(desc: core.BufferDescription) -> gym.Space[Any]:
    is_int = issubclass(desc.type.type, numbers.Integral)
    if is_int and desc.low == 0 and desc.high == 1:
        # CHANGED it does not respect the sensor dtype but instead uses int8
        # return gym.spaces.MultiBinary(desc.shape)
        pass
    if desc.categorical and is_int and len(desc.shape) == 1:
        return gym.spaces.MultiDiscrete(nvec=[int(desc.high - desc.low + 1)] *
                                        desc.shape[0],
                                        dtype=desc.type.type,
                                        start=[int(desc.low)] * desc.shape[0])
    return gym.spaces.Box(desc.low,
                          desc.high,
                          desc.shape,
                          dtype=desc.type.type)


def make_composed_space(
        value: Mapping[str, core.BufferDescription]) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        k: make_simple_space(desc)
        for k, desc in value.items()
    })


def get_relative_target_position(behavior: core.Behavior) -> core.Vector2:
    if behavior.target.position is None:
        return np.zeros(2, dtype=core.FloatType)
    return core.to_relative(behavior.target.position - behavior.position,
                            behavior.pose)


def get_space_for_sensor(sensor: sim.Sensor) -> gym.spaces.Dict:
    return make_composed_space(sensor.description)


def get_space_for_state(state: core.SensingState) -> gym.spaces.Dict:
    return make_composed_space({
        k: v.description
        for k, v in state.buffers.items()
    })


def get_state(behavior: core.Behavior | None,
              state: core.SensingState | None) -> core.SensingState | None:
    if state is None and behavior and isinstance(behavior.environment_state,
                                                 core.SensingState):
        return behavior.environment_state
    return state


class GymAgent:

    def __init__(self,
                 action: ActionProtocol,
                 observation: ObservationProtocol,
                 behavior: core.Behavior | None = None,
                 state: core.SensingState | None = None):
        state = get_state(behavior, state)
        if state:
            sensing_space = get_space_for_state(state)
        else:
            sensing_space = gym.spaces.Dict({})
        self.observation_config = copy.copy(observation)
        self.action_config = copy.copy(action)
        if behavior:
            self.action_config.configure(behavior)
        self.observation_config.configure(behavior, sensing_space)
        self.init(behavior, state)

    def is_configured(self, warn: bool = False) -> bool:
        return (self.action_config.is_configured(warn)
                and self.observation_config.is_configured(warn))

    @property
    def action_space(self) -> gym.Space[Any]:
        return self.action_config.space

    @property
    def observation_space(self) -> gym.Space[Any]:
        return self.observation_config.space

    def init(self, behavior: core.Behavior | None,
             state: core.SensingState | None) -> None:
        self._behavior = behavior
        state = get_state(behavior, state)
        if state:
            self._buffers = dict(state.buffers)
        else:
            self._buffers = {}
        history = self.observation_config.history
        if history > 1:
            self._stack: deque[dict[str, Array]
                               | Array] | None = deque(maxlen=history)
        else:
            self._stack = None

    def get_cmd_from_action(self, action: Array,
                            time_step: float) -> core.Twist2:
        return self.action_config.get_cmd_from_action(action, self._behavior,
                                                      time_step)

    def get_action(self, time_step: float) -> Array:
        assert self._behavior is not None
        return self.action_config.get_action(self._behavior, time_step)

    def update_observation(self) -> dict[str, Array] | Array:
        fs = self.observation_config.get_observation(self._behavior,
                                                     self._buffers)
        if self._stack is None:
            return fs
        self._stack.append(fs)
        while len(self._stack) < self.observation_config.history:
            self._stack.append(fs)
        return np.asarray(self._stack)

    # def reset(self) -> None:
    #     if self._stack:
    #         self._stack.clear()

    def __getstate__(self) -> tuple[Any, ...]:
        return (self.action_config, self.observation_config)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.action_config, self.observation_config) = state
        self.init(None, None)


@dc.dataclass
class Agent:
    gym: GymAgent | None = None
    reward: Reward | None = None
    navground: sim.Agent | None = None
    state: core.SensingState | None = None
    sensor: sim.Sensor | None = None
    policy: Any = None

    def get_sensor(self) -> sim.Sensor | None:
        if self.sensor:
            return self.sensor
        if self.navground and isinstance(self.navground.state_estimation,
                                         sim.Sensor):
            return self.navground.state_estimation
        return None

    def update_state(self, world: sim.World) -> None:
        if self.sensor and self.state and self.navground:
            self.sensor.update(self.navground, world, self.state)

    def is_configured(self, warn: bool = False) -> bool:
        if self.gym:
            return self.gym.is_configured(warn)
        if warn:
            warnings.warn("No Gymnasium configuration", stacklevel=1)
        return False


def create_agents_in_group(
        world: sim.World,
        group: GroupConfig,
        max_number_of_agents: int | None = None) -> dict[int, Agent]:
    agents = {}
    sensor = group.get_sensor()
    world_agents = group.indices.sub_sequence(world.agents)
    if world_agents:
        representative_behavior = world_agents[0].behavior
    else:
        representative_behavior = None
    if max_number_of_agents is not None and max_number_of_agents > len(
            world.agents):
        number_of_agents = max_number_of_agents
    else:
        number_of_agents = len(world.agents)
    indices = group.indices.as_set(number_of_agents)
    for i in indices:
        if sensor:
            state = core.SensingState()
            sensor.prepare_state(state)
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
        if ng_agent and group.color:
            ng_agent.color = group.color
        if ng_agent and group.tag:
            ng_agent.add_tag(group.tag)

        agents[i] = Agent(state=state,
                          gym=gym_agent,
                          sensor=sensor,
                          reward=group.reward,
                          navground=ng_agent)
        # TODO(Jerome):
        # Set policy
    return agents


def create_agents_in_groups(
        world: sim.World,
        groups: Collection[GroupConfig],
        max_number_of_agents: int | None = None) -> dict[int, Agent]:
    agents = {}
    for group in groups:
        agents.update(
            create_agents_in_group(world, group, max_number_of_agents))
    return agents
