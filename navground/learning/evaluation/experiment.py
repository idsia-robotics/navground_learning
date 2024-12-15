from __future__ import annotations

import copy
from collections.abc import Collection
from functools import partial
from typing import TYPE_CHECKING

from navground import sim

from ..config import GroupConfig, merge_groups_configs
from ..env import BaseEnv
from ..internal.base_env import NavgroundBaseEnv
from ..parallel_env import BaseParallelEnv
from ..probes.reward import RewardProbe
from ..types import AnyPolicyPredictor, Bounds, PathLike, Reward
from .scenario import InitPolicyBehavior

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def make_experiment(scenario: sim.Scenario,
                    groups: Collection[GroupConfig] = tuple(),
                    reward: Reward | None = None,
                    record_reward: bool = True,
                    policy: AnyPolicyPredictor | PathLike = '',
                    bounds: Bounds | None = None,
                    terminate_outside_bounds: bool = True,
                    deterministic: bool = True) -> sim.Experiment:
    """
    Initializes an navground experiment where groups of agents
    are configured with possibly different policies and
    the rewards is optionally recorded.

    If ``groups`` is not empty, it make a copy of the scenario, to which it adds
    adds :py:class:`navground.learning.evaluation.scenario.InitPolicyBehavior`
    to initialize the groups.

    If ``record_reward`` is set, it adds a
    :py:class:`navground.learning.probes.reward.RewardProbe`.

    :param  scenario:                  The scenario
    :param  groups:                    The configuration of the groups
    :param  reward:                    The default reward to record
                                       (when not specified in the group config)
    :param  record_reward:             Whether to record the rewards
    :param  policy:                    The default policy
                                       (when not specified in the group config)
    :param  bounds:                    Optional termination boundaries
    :param  terminate_outside_bounds:  Whether to terminate
                                       if some of the agents exits the boundaries
    :param  deterministic:             Whether to apply the policies deterministically

    :returns:   The experiment
    """
    experiment = sim.Experiment()
    if not groups:
        groups = [GroupConfig(policy=policy)]
    if groups:
        experiment.scenario = copy.copy(scenario)
        init = InitPolicyBehavior(
            groups=groups,
            bounds=bounds,
            terminate_outside_bounds=terminate_outside_bounds,
            deterministic=deterministic)
        experiment.scenario.add_init(init)
    else:
        experiment.scenario = scenario
    if record_reward:
        experiment.add_record_probe(
            "reward", partial(RewardProbe, groups=groups, reward=reward))
    return experiment


def make_experiment_with_env(env: BaseEnv | BaseParallelEnv | VecEnv,
                             groups: Collection[GroupConfig] = tuple(),
                             policy: AnyPolicyPredictor | PathLike = '',
                             reward: Reward | None = None,
                             record_reward: bool = True,
                             deterministic: bool = True) -> sim.Experiment:
    """
    Similar to :py:func:`make_experiment` but using the configuration stored in
    an environment:
    ``groups`` are merged using :py:func:`navground.learning.config.merge_groups_configs`.


    :param  env:                       The environment
    :param  groups:                    The configuration of the groups
    :param  reward:                    The default reward to record
                                       (when not specified in the group config)
    :param  record_reward:             Whether to record the rewards
    :param  policy:                    The default policy
                                       (when not specified in the group config)
    :param  bounds:                    Optional termination boundaries
    :param  terminate_outside_bounds:  Whether to terminate
                                       if some of the agents exits the boundaries
    :param  deterministic:             Whether to apply the policies deterministically


    :returns:   The experiment
    """
    from stable_baselines3.common.vec_env import VecEnv

    if not groups:
        groups = [GroupConfig(policy=policy)]
    if isinstance(env, VecEnv):
        scenario: sim.Scenario | None = env.get_attr('_scenario')[0]
        max_duration: float = env.get_attr('max_duration')[0]
        time_step: float = env.get_attr('time_step')[0]
        env_groups = env.get_attr('groups_config')[0]
        possible_agents = env.get_attr('_possible_agents')[0]
    else:
        if not isinstance(env.unwrapped, NavgroundBaseEnv):
            raise TypeError(
                f"Environment {env} is not a Navground environment.")
        scenario = env.unwrapped._scenario
        max_duration = env.unwrapped.max_duration
        time_step = env.unwrapped.time_step
        env_groups = env.unwrapped.groups_config
        possible_agents = env.unwrapped._possible_agents

    experiment = sim.Experiment()
    if scenario:
        experiment.scenario = copy.copy(scenario)
    if max_duration > 0:
        experiment.steps = int(max_duration / time_step)
    init = InitPolicyBehavior.with_env(env=env,
                                       groups=groups,
                                       deterministic=deterministic)
    experiment.scenario.add_init(init)
    groups = merge_groups_configs(groups, env_groups, len(possible_agents))
    if record_reward:
        experiment.add_record_probe(
            "reward", partial(RewardProbe, groups=groups, reward=reward))
    return experiment
