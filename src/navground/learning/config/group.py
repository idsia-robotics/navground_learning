from __future__ import annotations

import dataclasses as dc
from collections.abc import Collection, Mapping
from typing import Any, cast

import yaml
from navground import sim

from ..indices import Indices
from ..types import (AnyPolicyPredictor, GroupObservationsTransform, JSONAble,
                     ObservationTransform, PathLike, Reward, SensorLike,
                     SensorSequenceLike, TerminationCondition)
from .base import ActionConfig, ObservationConfig


def load_sensor(value: str) -> sim.Sensor | None:
    se = sim.load_state_estimation(value)
    if isinstance(se, sim.Sensor):
        return se
    print(f"Invalid sensor {se}")
    return None


def load_sensors(value: str) -> list[sim.Sensor]:
    ds = yaml.safe_load(value)
    ses = (sim.load_sensor(yaml.dump(d)) for d in ds)
    return [se for se in ses if se]


def make_sensor(value: SensorLike | None) -> sim.Sensor | None:
    if isinstance(value, dict):
        value = yaml.dump(value)
    if isinstance(value, str):
        value = load_sensor(value)
    return value


def make_sensors(value: SensorSequenceLike) -> list[sim.Sensor]:
    if isinstance(value, str):
        return load_sensors(value)
    return [x for x in (make_sensor(se) for se in value) if x]


def get_sensor_as_dict(sensor: SensorLike | None) -> dict[str, Any]:
    if sensor is None:
        return {}
    if isinstance(sensor, dict):
        return sensor
    if isinstance(sensor, sim.Sensor):
        sensor = sim.dump(sensor)
    return cast("dict[str, Any]", yaml.safe_load(sensor))


@dc.dataclass
class GroupConfig:
    """
    Configure a group of navground agents so that they
    are exposed in a :py:class:`gymnasium.Env` (only single agents)
    or :py:class:`pettingzoo.utils.env.ParallelEnv` (one or more agents).

    Group configurations can be stuck. For instance, the configuration provided
    to define the environment for training needs needs to define at least
    :py:attr:`indices`, :py:attr:`action` and
    :py:attr:`observation`, and possibly also one :py:attr:`sensors`.
    Instead, for evaluation, we may provide just the policy,
    as the agents are already configured.

    :param indices: The indices of the agents
                    in the :py:attr:`navground.sim.World.agents` list.

    :param action: The actions configuration.

    :param observation: The observations configuration.

    :param sensor: An optional sensor that will be added to :py:obj:`sensors`.

    :param sensors: A sequence of sensor to generate observations for the agents
                    or its YAML representation. If
                    Items of class :py:class:`str` will be interpreted as the YAML
                    representation of a sensor.
                    Items of class :py:class:`dict` will be dumped to YAML and
                    then treated as a :py:class:`str`.
                    If empty, it will use the agents' own sensors.

    :param reward: An optional reward function to use.

    :param color : An optional color for the agents (only used for displaying)

    :param tag : An optional tag added to the agents :py:attr:`navground.sim.Agent.tags`,
                 (does not impact simulation, only used to identify the agents)

    :param policy: The policy assigned to the agents during evaluation.

    :param deterministic: Whether the agents apply such policy deterministically.

    :param terminate_on_success: Whether to terminate when the agent succeeds

    :param terminate_on_failure: Whether to terminate when the agent fails

    :param success_condition: Optional success condition

    :param failure_condition: Optional failure condition

    :param grouped: Whether the policy is grouped

    :param pre: An optional transformation to apply to observations

    :param group_pre: An optional transformation to apply to group observations
    """
    indices: Indices = Indices.all()
    """The indices of the agents in the :py:attr:`navground.sim.World.agents` list"""
    action: ActionConfig | None = None
    """The actions configuration"""
    observation: ObservationConfig | None = None
    """The observations configuration"""
    sensor: dc.InitVar[SensorLike | None] = None
    sensors: SensorSequenceLike = dc.field(default_factory=list)
    """List of sensors to generate observations for the agents"""
    reward: Reward | None = None
    """The reward function"""
    color: str = ''
    """The agents' color"""
    tag: str = ''
    """An optional additional tag for the agents"""
    policy: AnyPolicyPredictor | PathLike = ''
    """The policy assigned to the agents (during evaluation)"""
    deterministic: bool = False
    """Whether the agents apply such policy deterministically"""
    terminate_on_success: bool | None = None
    """Whether to terminate when the agent succeeds"""
    terminate_on_failure: bool | None = None
    """Whether to terminate when the agent fails"""
    success_condition: TerminationCondition | None = None
    """Success condition"""
    failure_condition: TerminationCondition | None = None
    """Failure condition"""
    grouped: bool | None = None
    """Whether the policy is grouped"""
    pre: ObservationTransform | None = None
    """An optional transformation to apply to observations of individual agents"""
    group_pre: GroupObservationsTransform | None = None
    """An optional transformation to apply to the whole group observations"""

    def __post_init__(self, sensor: SensorLike | None) -> None:
        self.indices = Indices(self.indices)
        self.sensors = make_sensors(self.sensors)
        sensor = make_sensor(sensor)
        if sensor:
            if not isinstance(self.sensors, list):
                self.sensors = []
            self.sensors.append(sensor)

    @property
    def asdict(self) -> dict[str, Any]:
        """
        A JSON-able representation of the configuration

        :returns:  A JSON-able dict
        """
        rs: dict[str, Any] = {}
        if self.color:
            rs['color'] = self.color
        if self.tag:
            rs['tag'] = self.tag
        if self.deterministic:
            rs['deterministic'] = self.deterministic
        if self.observation:
            rs['observation'] = self.observation.asdict
        if self.action:
            rs['action'] = self.action.asdict
        if self.reward:
            rs['reward'] = self.reward.asdict
        if self.indices:
            rs['indices'] = self.indices.asdict
        if self.grouped:
            rs['grouped'] = self.grouped
        rs['terminate_on_success'] = self.terminate_on_success
        rs['terminate_on_failure'] = self.terminate_on_failure
        rs['sensors'] = [get_sensor_as_dict(sensor) for sensor in self.sensors]
        return rs

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GroupConfig:
        loadables: dict[str, type] = {
            'observation': ObservationConfig,
            'action': ActionConfig,
            'reward': Reward,
            'indices': Indices,
        }
        kwargs: dict[str, Any] = {
            k: value[k]
            for k in ('color', 'tag', 'deterministic', 'sensors', 'sensor',
                      'terminate_on_success', 'terminate_on_failure', 'grouped')
            if k in value
        }
        kwargs.update((k, cast("JSONAble", item_cls).from_dict(value[k]))
                      for k, item_cls in loadables.items() if k in value)
        return GroupConfig(**kwargs)

    def get_sensors(self) -> list[sim.Sensor]:
        return make_sensors(self.sensors)


def get_first_reward(groups: Collection[GroupConfig]) -> Reward | None:
    for group in groups:
        if group.reward:
            return group.reward
    return None


def merge_groups_configs(groups: Collection[GroupConfig],
                         other_groups: Collection[GroupConfig],
                         max_number: int) -> list[GroupConfig]:
    """
    Overlaps two sets of group configurations, letting the first
    inherit attributes from the second if they are not specified.

    :param      groups:        A collection of groups configuration
    :param      other_groups:  Another collection of groups configuration to be inherited from.
    :param      max_number:    The maximum number of agents, needed to intersect
                               the groups :py:attr:`GroupConfig.indices` to identify
                               which agents belong to groups in both sets.
    :returns:   A new list of groups, where the attributes from both groups collections.
    """
    gs: list[GroupConfig] = []
    for g1 in groups:
        for g2 in other_groups:
            indices = g1.indices.intersect(g2.indices, max_number)
            if indices:
                g = dc.replace(g1, indices=indices)
                if g1.action is None:
                    g.action = g2.action
                if not g1.sensors:
                    g.sensors = g2.sensors
                if g1.reward is None:
                    g.reward = g2.reward
                if g1.observation is None:
                    g.observation = g2.observation
                if g1.color is None:
                    g.color = g2.color
                if g1.tag is None:
                    g.tag = g2.tag
                if g1.policy is None:
                    g.policy = g2.policy
                if g1.terminate_on_success is None:
                    g.terminate_on_success = g2.terminate_on_success
                if g1.terminate_on_failure is None:
                    g.terminate_on_failure = g2.terminate_on_failure
                if g1.success_condition is None:
                    g.success_condition = g2.success_condition
                if g1.failure_condition is None:
                    g.failure_condition = g2.failure_condition
                if g1.grouped is None:
                    g.grouped = g2.grouped
                if g1.pre is None:
                    g.pre = g2.pre
                if g1.group_pre is None:
                    g.group_pre = g2.group_pre
                gs.append(g)
    return gs
