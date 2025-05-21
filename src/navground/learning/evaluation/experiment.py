from __future__ import annotations

import copy
from collections.abc import Collection
from functools import partial
from typing import TYPE_CHECKING

from navground import sim

from ..config import GroupConfig, merge_groups_configs
from ..env import BaseEnv
from ..indices import join_indices
from ..internal.base_env import NavgroundBaseEnv
from ..parallel_env import BaseParallelEnv
from ..probes.reward import RewardProbe
from ..probes.success import SuccessProbe
from ..types import (AnyPolicyPredictor, Bounds, ObservationTransform,
                     PathLike, Reward)
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
    experiment.terminate_when_all_idle_or_stuck = False
    if not groups:
        groups = [GroupConfig(policy=policy)]
    if groups:
        experiment.scenario = copy.deepcopy(scenario)
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
                             record_success: bool = True,
                             deterministic: bool = True,
                             grouped: bool = False,
                             pre: ObservationTransform | None = None) -> sim.Experiment:
    """
    Similar to :py:func:`make_experiment` but using the configuration stored in
    an environment:
    ``groups`` are merged using :py:func:`navground.learning.config.merge_groups_configs`.


    :param  env:                       The environment
    :param  groups:                    The configuration of the groups
    :param  reward:                    The default reward to record
                                       (when not specified in the group config)
    :param  record_reward:             Whether to record the rewards
    :param  record_success:             Whether to record the success
    :param  policy:                    The default policy
                                       (when not specified in the group config)
    :param  grouped:                   Whether the policy is grouped.
    :param  deterministic:             Whether to apply the policies deterministically
    :param  pre:                       An optional transformation to apply to observations

    :returns:   The experiment
    """
    from stable_baselines3.common.vec_env import VecEnv

    if not groups:
        groups = [GroupConfig(policy=policy)]
    # Should not be anymore necessary as `get_attr` is added in `make_vec_from_penv`
    # if isinstance(env, VecEnv):
    #     try:
    #         env = env.unwrapped.vec_envs[0].par_env
    #     except Exception:
    #         pass
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
    experiment.terminate_when_all_idle_or_stuck = False
    if scenario:
        experiment.scenario = copy.deepcopy(scenario)
    if max_duration > 0:
        experiment.steps = int(max_duration / time_step)
    init = InitPolicyBehavior.with_env(env=env,
                                       groups=groups,
                                       deterministic=deterministic,
                                       grouped=grouped,
                                       pre=pre)
    experiment.scenario.add_init(init)
    groups = merge_groups_configs(groups, env_groups, len(possible_agents))
    if record_success:
        indices = join_indices(
            (group.indices for group in groups
             if group.success_condition or group.failure_condition),
            len(possible_agents))
        if indices:
            experiment.add_record_probe("success",
                                        SuccessProbe.with_indices(indices))
    if record_reward:
        experiment.add_record_probe(
            "reward", partial(RewardProbe, groups=groups, reward=reward))
    return experiment
