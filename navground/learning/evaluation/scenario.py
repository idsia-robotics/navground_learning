from __future__ import annotations

import warnings
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

from navground import core, sim

from ..behaviors.policy import PolicyBehavior
from ..config import (ControlActionConfig, DefaultObservationConfig,
                      GroupConfig, merge_groups_configs)
from ..env import BaseEnv
from ..indices import Indices, IndicesLike
from ..internal.base_env import NavgroundBaseEnv
from ..parallel_env import BaseParallelEnv
from ..types import AnyPolicyPredictor, Bounds, PathLike

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def set_policy_behavior(world: sim.World,
                        indices: IndicesLike,
                        policy: AnyPolicyPredictor | PathLike,
                        action_config: ControlActionConfig,
                        observation_config: DefaultObservationConfig,
                        sensor: sim.Sensor | None = None,
                        deterministic: bool = True) -> set[int]:
    indices = Indices(indices)
    all_indices = indices.as_set(len(world.agents))
    for i in all_indices:
        agent = world.agents[i]
        if agent.behavior:
            agent.behavior = PolicyBehavior.clone_behavior(
                behavior=agent.behavior,
                policy=policy,
                action_config=action_config,
                observation_config=observation_config,
                deterministic=deterministic)
            if sensor:
                agent.state_estimation = sensor
        else:
            all_indices.remove(i)
    return set(all_indices)


def set_policy_behavior_with_env(world: sim.World,
                                 env: BaseEnv | BaseParallelEnv | VecEnv,
                                 policy: AnyPolicyPredictor | PathLike,
                                 indices: IndicesLike = "ALL",
                                 deterministic: bool = True) -> set[int]:
    indices = Indices(indices)
    if not isinstance(env.unwrapped, NavgroundBaseEnv):
        warnings.warn(f"Environment {env} is not a Navground environment.",
                      stacklevel=1)
        return set()
    indices_set = indices.as_set(len(world.agents))
    all_indices: set[int] = set()
    for group in env.unwrapped.groups_config:
        group_indices = group.indices.as_set(len(world.agents)) & indices_set
        if group_indices and isinstance(
                group.action, ControlActionConfig) and isinstance(
                    group.observation, DefaultObservationConfig):
            set_policy_behavior(world,
                                policy=policy,
                                indices=Indices(group_indices),
                                action_config=group.action,
                                observation_config=group.observation,
                                sensor=group.get_sensor(),
                                deterministic=deterministic)
            all_indices |= group_indices
    return all_indices


def to_bounds(bb: sim.BoundingBox) -> Bounds:
    return bb.p1, bb.p2


def is_outside(p: core.Vector2, bounds: Bounds) -> bool:
    return any(p < bounds[0]) or any(p > bounds[1])


def is_any_agents_outside(
    bounds: Bounds | None = None,
    agent_indices: Indices = Indices({0})
) -> Callable[[sim.World], bool]:

    def f(world: sim.World) -> bool:
        nonlocal bounds
        if bounds is None:
            bounds = to_bounds(world.bounding_box)
        return any(
            is_outside(agent.position, bounds)
            for agent in agent_indices.sub_sequence(world.agents))

    return f


# pickle does not work with functions factories
# so better to write it has a callable class
class InitPolicyBehavior:
    """
    A navground scenario initializer to configure groups of agents.

    It is designed to be added to a scenario, like

    >>> from navground import sim
    >>> from navground.learning import GroupConfig
    >>>
    >>> scenario = sim.load_scenario(...)
    >>> groups = [GroupConfig(policy='policy.onnx', color='red', indices='ALL')]
    >>> scenario.add_init(InitPolicyBehavior(groups=groups))
    >>> world = scenario.make_world(seed=101)
    >>> another_world = scenario.make_world(seed=313)

    :param groups: The configuration of groups of agents
    :param bounds: Optional termination boundaries
    :param terminate_outside_bounds: Whether to terminate
        if some of the agents exits the boundaries
    :param deterministic: Whether to apply the policies deterministically

    """

    def __init__(self,
                 groups: Collection[GroupConfig] = tuple(),
                 bounds: Bounds | None = None,
                 terminate_outside_bounds: bool = True,
                 deterministic: bool = True) -> None:
        self.groups = groups
        self.bounds = bounds
        self.terminate_outside_bounds = terminate_outside_bounds
        self.deterministic = deterministic

    def __call__(self, world: sim.World, seed: int | None = None) -> None:
        """
        Configures the policy, sensor, color, ... of the agents according
        to their group.

        :param      world:  The world
        :param      seed:   The random seed
        """
        agent_indices = set()
        for group in self.groups:
            if group.policy and isinstance(
                    group.action, ControlActionConfig) and isinstance(
                    group.observation, DefaultObservationConfig):
                group_indices = set_policy_behavior(
                    world,
                    action_config=group.action,
                    observation_config=group.observation,
                    sensor=group.get_sensor(),
                    policy=group.policy,
                    indices=group.indices,
                    deterministic=self.deterministic)
                agent_indices |= group_indices
                for i in group_indices:
                    if group.color:
                        world.agents[i].color = group.color
                    if group.tag:
                        world.agents[i].add_tag(group.tag)
        if agent_indices:
            if self.terminate_outside_bounds:
                tc = is_any_agents_outside(self.bounds, Indices(agent_indices))
                world.set_termination_condition(tc)

    @classmethod
    def with_env(cls,
                 env: BaseEnv | BaseParallelEnv | VecEnv,
                 groups: Collection[GroupConfig] = tuple(),
                 bounds: Bounds | None = None,
                 terminate_outside_bounds: bool | None = None,
                 deterministic: bool = True) -> InitPolicyBehavior:
        """
        Returns a scenario initializer using the configuration stored in
        an environment.

        groups are merged using :py:func:`navground.learning.config.merge_groups_configs`.

        :param env:    The environment
        :param groups:      The configuration of groups of agents
        :param bounds:      Optional termination boundaries
        :param terminate_outside_bounds: Whether to terminate if some of
            the agents exits the boundaries
        :param deterministic: Whether to apply the policies deterministically

        :returns:   The scenario initializer.
        """
        from stable_baselines3.common.vec_env import VecEnv

        if isinstance(env, VecEnv):
            if not bounds:
                bounds = env.get_attr('bounds')[0]
            if terminate_outside_bounds is None:
                terminate_outside_bounds = env.get_attr(
                    'terminate_outside_bounds')[0]
            env_groups = env.get_attr('groups_config')[0]
            possible_agents = env.get_attr('_possible_agents')[0]
        else:
            if not isinstance(env.unwrapped, NavgroundBaseEnv):
                raise TypeError(
                    f"Environment {env} is not a Navground environment.")
            if not bounds:
                bounds = env.unwrapped.bounds
            if terminate_outside_bounds is None:
                terminate_outside_bounds = env.unwrapped.terminate_outside_bounds
            env_groups = env.unwrapped.groups_config
            possible_agents = env.unwrapped._possible_agents
        if terminate_outside_bounds is None:
            terminate_outside_bounds = False
        groups = merge_groups_configs(groups, env_groups, len(possible_agents))
        return cls(groups, bounds, terminate_outside_bounds, deterministic)
