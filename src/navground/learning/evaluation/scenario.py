from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import TYPE_CHECKING, SupportsInt

from navground import core, sim

from ..behaviors import GroupedPolicyBehavior, PolicyBehavior
from ..config import (ControlActionConfig, DefaultObservationConfig,
                      GroupConfig, merge_groups_configs)
from ..env import BaseEnv
from ..indices import Indices, IndicesLike
from ..internal.base_env import NavgroundBaseEnv
from ..parallel_env import BaseParallelEnv
from ..types import (AnyPolicyPredictor, Bounds, GroupObservationsTransform,
                     ObservationTransform, PathLike)

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

AnyPolicyBehavior = PolicyBehavior | GroupedPolicyBehavior


def set_policy_behavior(
        world: sim.World,
        indices: IndicesLike,
        policy: AnyPolicyPredictor | PathLike,
        action_config: ControlActionConfig,
        observation_config: DefaultObservationConfig,
        sensors: Sequence[sim.Sensor] = tuple(),
        deterministic: bool = True,
        grouped: bool = False,
        pre: ObservationTransform | None = None,
        group_pre: GroupObservationsTransform | None = None) -> set[int]:
    indices = Indices(indices)
    all_indices = indices.as_set(len(world.agents))
    if grouped:
        behavior_cls: type[AnyPolicyBehavior] = GroupedPolicyBehavior
    else:
        behavior_cls = PolicyBehavior
    for i in all_indices:
        agent = world.agents[i]
        if agent.behavior:
            agent.behavior = behavior_cls.clone_behavior(
                behavior=agent.behavior,
                policy=policy,
                action_config=action_config,
                observation_config=observation_config,
                deterministic=deterministic,
                pre=pre)
            if group_pre and isinstance(agent.behavior, GroupedPolicyBehavior):
                agent.behavior.set_group_pre(group_pre)
            if sensors:
                agent.state_estimations = list(sensors)
        else:
            all_indices.remove(i)
    return set(all_indices)


def set_policy_behavior_with_env(
        world: sim.World,
        env: BaseEnv | BaseParallelEnv | VecEnv,
        policy: AnyPolicyPredictor | PathLike,
        indices: IndicesLike = "ALL",
        deterministic: bool = True,
        grouped: bool = False,
        pre: ObservationTransform | None = None,
        group_pre: GroupObservationsTransform | None = None) -> set[int]:
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
                                sensors=group.get_sensors(),
                                deterministic=deterministic,
                                grouped=grouped,
                                pre=pre,
                                group_pre=group_pre)
            all_indices |= group_indices
    return all_indices


def is_outside(p: core.Vector2, bounds: Bounds) -> bool:
    return any(p < bounds[0]) or any(p > bounds[1])


def is_any_agents_outside(
    bounds: Bounds | None = None,
    agent_indices: Indices = Indices({0})
) -> Callable[[sim.World], bool]:

    def f(world: sim.World) -> bool:
        nonlocal bounds
        if bounds is None:
            bounds = world.bounds
        return any(
            is_outside(agent.position, bounds)
            for agent in agent_indices.sub_sequence(world.agents))

    return f


def is_outside_world_bounds(
        world: sim.World,
        bounds: Bounds | None = None
) -> Callable[[sim.Agent, sim.World], bool]:
    if bounds is None:
        bounds = world.bounds

    def f(agent: sim.Agent, world: sim.World) -> bool:
        return is_outside(agent.position, bounds)

    return f


Condition = tuple[Indices, Callable[[sim.Agent, sim.World], bool], bool | None]


def make_termination_condition(
        any_of: Iterable[Condition], all_of: Iterable[Condition],
        not_terminal: Iterable[Condition]) -> Callable[[sim.World], bool]:

    def f(world: sim.World) -> bool:
        value = False
        for indices, fn, result in any_of:
            for agent in indices.sub_sequence(world.agents):
                if fn(agent, world):
                    value = True
                    if not hasattr(agent, '_success'):
                        agent._success = result  # type: ignore[attr-defined]
        c_value = None
        for indices, fn, result in all_of:
            for agent in indices.sub_sequence(world.agents):
                if fn(agent, world):
                    if c_value is None:
                        c_value = True
                    if not hasattr(agent, '_success'):
                        agent._success = result  # type: ignore[attr-defined]
                else:
                    c_value = False
        if c_value:
            value = True
        for indices, fn, result in not_terminal:
            for agent in indices.sub_sequence(world.agents):
                if not hasattr(agent, '_success') and fn(agent, world):
                    agent._success = result  # type: ignore[attr-defined]
        return value

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
    :param  deterministic: Whether to apply the policies deterministically
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

    def __call__(self, world: sim.World, seed: SupportsInt | None = None) -> None:
        """
        Configures the policy, sensor, color, ... of the agents according
        to their group.

        :param      world:  The world
        :param      seed:   The random seed
        """
        agent_indices: set[int] = set()
        all_of: list[Condition] = []
        any_of: list[Condition] = []
        not_terminal: list[Condition] = []
        for group in self.groups:
            if group.policy and isinstance(
                    group.action, ControlActionConfig) and isinstance(
                        group.observation, DefaultObservationConfig):
                group_indices = set_policy_behavior(
                    world,
                    action_config=group.action,
                    observation_config=group.observation,
                    sensors=group.get_sensors(),
                    policy=group.policy,
                    indices=group.indices,
                    deterministic=self.deterministic,
                    grouped=group.grouped or False,
                    pre=group.pre,
                    group_pre=group.group_pre)
                agent_indices |= group_indices
                for i in group_indices:
                    if group.color:
                        world.agents[i].color = group.color
                    if group.tag:
                        world.agents[i].add_tag(group.tag)
                if group.success_condition:
                    c = Indices(group_indices), group.success_condition, True
                    if group.terminate_on_success:
                        all_of.append(c)
                    else:
                        not_terminal.append(c)
                if group.failure_condition:
                    c = Indices(group_indices), group.failure_condition, False
                    if group.terminate_on_failure:
                        any_of.append(c)
                    else:
                        not_terminal.append(c)

        if agent_indices and self.terminate_outside_bounds:
            any_of.append((Indices(agent_indices),
                           is_outside_world_bounds(world, self.bounds), None))
        if all_of or any_of or not_terminal:
            tc = make_termination_condition(all_of=all_of,
                                            any_of=any_of,
                                            not_terminal=not_terminal)
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
        :param  deterministic: Whether to apply the policies deterministically

        :returns:   The scenario initializer.
        """
        from stable_baselines3.common.vec_env import VecEnv

        if isinstance(env, VecEnv):
            if not bounds:
                bounds = env.get_attr('bounds')[0]
            if terminate_outside_bounds is None:
                terminate_outside_bounds = env.get_attr(
                    'truncate_outside_bounds')[0]
            env_groups = env.get_attr('groups_config')[0]
            possible_agents = env.get_attr('_possible_agents')[0]
        else:
            if not isinstance(env.unwrapped, NavgroundBaseEnv):
                raise TypeError(
                    f"Environment {env} is not a Navground environment.")
            if not bounds:
                bounds = env.unwrapped.bounds
            if terminate_outside_bounds is None:
                terminate_outside_bounds = env.unwrapped.truncate_outside_bounds
            env_groups = env.unwrapped.groups_config
            possible_agents = env.unwrapped._possible_agents
        if terminate_outside_bounds is None:
            terminate_outside_bounds = False
        groups = merge_groups_configs(groups, env_groups, len(possible_agents))
        return cls(groups, bounds, terminate_outside_bounds, deterministic)
