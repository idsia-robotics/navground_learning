from __future__ import annotations

import dataclasses as dc
from collections.abc import Collection, Mapping
from typing import Any, cast

import yaml
from navground import sim

from ..indices import Indices
from ..types import AnyPolicyPredictor, PathLike, Reward, JSONAble
from .base import ActionConfig, ObservationConfig


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
    return value


def get_sensor_as_dict(
        sensor: sim.Sensor | None | str | dict[str, Any]) -> dict[str, Any]:
    if sensor is None:
        return {}
    if isinstance(sensor, dict):
        return sensor
    if isinstance(sensor, sim.Sensor):
        sensor = sim.dump(sensor)
    return cast(dict[str, Any], yaml.safe_load(sensor))


@dc.dataclass
class GroupConfig:
    """
    Configure a group of navground agents so that they
    are exposed in a :py:class:`gymnasium.Env` (only single agents)
    or :py:class:`pettingzoo.utils.env.ParallelEnv` (one or more agents).

    Group configurations can be stuck. For instance, the configuration provided
    to define the environment for training needs needs to define at least
    :py:attr:`indices`, :py:attr:`action` and
    :py:attr:`observation`, and possibly also the :py:attr:`sensor`.
    Instead, for evaluation, we may provide just the policy,
    as the agents are already configured.

    :param indices: The indices of the agents
                    in the :py:attr:`navground.sim.World.agents` list.

    :param action: The actions configuration.

    :param observation: The observations configuration.

    :param sensor: A sensor to produce observations for the agents.
                   If a :py:class:`str`, it will be interpreted as the YAML
                   representation of a sensor.
                   If a :py:class:`dict`, it will be dumped to YAML and
                   then treated as a :py:class:`str`.
                   If None, it will use the agents' own state estimation, if a sensor.

    :param reward: An optional reward function to use.

    :param color : An optional color for the agents (only used for displaying)

    :param tag : An optional tag added to the agents :py:attr:`navground.sim.Agent.tags`,
                 (does not impact simulation, only used to identify the agents)

    :param policy: The policy assigned to the agents during evaluation.

    :param deterministic: Whether the agents apply such policy deterministically.

    """
    indices: Indices = Indices.all()
    """The indices of the agents in the :py:attr:`navground.sim.World.agents` list"""
    action: ActionConfig | None = None
    """The actions configuration"""
    observation: ObservationConfig | None = None
    """The observations configuration"""
    sensor: sim.Sensor | str | dict[str, Any] | None = None
    """A sensor to produce observations for the agents"""
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

    def __post_init__(self) -> None:
        self.indices = Indices(self.indices)

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
        rs['sensor'] = get_sensor_as_dict(self.sensor)
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
            for k in ('color', 'tag', 'deterministic', 'sensor') if k in value
        }
        kwargs.update((k, cast(JSONAble, item_cls).from_dict(value[k]))
                      for k, item_cls in loadables.items() if k in value)
        return GroupConfig(**kwargs)

    def get_sensor(self) -> sim.Sensor | None:
        return make_sensor(self.sensor)


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
                if g1.sensor is None:
                    g.sensor = g2.sensor
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
                gs.append(g)
    return gs
