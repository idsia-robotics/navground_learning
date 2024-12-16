from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from navground import sim
from pettingzoo.utils.env import ParallelEnv

from ..config import GroupConfig
from ..env import BaseEnv
from ..indices import Indices, IndicesLike
from ..parallel_env import BaseParallelEnv, MultiAgentNavgroundEnv
from ..types import Reward
from .evaluate_policies import evaluate_policies
from .evaluate_policy import evaluate_policy
from .experiment import make_experiment, make_experiment_with_env

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def evaluate(
    env: BaseEnv | BaseParallelEnv | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    return_episode_rewards: bool = False,
    warn: bool = True,
    indices: IndicesLike = Indices.all(),
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Similar interface as StableBaseline3
    :py:func:`stable_baselines3.common.evaluation.evaluate_policy`
    to evaluate the navground policy/behavior.

    Internally, it runs :py:func:`navground.learning.evaluation.evaluate_policy`
    or :py:func:`navground.learning.evaluation.evaluate_policies`,
    depending if ``env`` is a single or a multi-agent environment.

    :param      env:                     The environment
    :param      n_eval_episodes:         The number of episodes
    :param      deterministic:           Whether the policy is applied deterministically
    :param      return_episode_rewards:  Whether to return all episodes (vs averaging them)
    :param      warn:                    Whether to enable warnings
    :param      indices:                 The indices of the agents whose
                                         reward we want to record

    :returns:   Same as :py:func:`stable_baselines3.common.evaluation.evaluate_policy` return,
       a tuple of (mean, std dev) or a tuple
       [reward_ep_1, reward_ep_2, ...], [length_ep_1, length_ep_2, ...]
    """
    from stable_baselines3.common.vec_env import VecEnv

    if isinstance(env, ParallelEnv):
        # TODO(Jerome): make it general for heterogeneous envs
        model = cast(MultiAgentNavgroundEnv,
                     env).get_policy(env.possible_agents[0])
        models = [(indices, model)]
        result = evaluate_policies(
            env=env,
            models=models,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=return_episode_rewards,
            warn=warn)
        if not return_episode_rewards:
            means, stds = cast(tuple[list[float], list[float]], result)
            return means[0], stds[0]
        grs, ls, gns = cast(
            tuple[list[list[float]], list[int], list[list[int]]], result)
        return [value / n for value, n in zip(grs[0], gns[0], strict=True)], ls
    if isinstance(env, VecEnv):
        model = env.get_attr("policy", [0])[0]
    else:
        model = env.get_wrapper_attr("policy")
    return evaluate_policy(env=env,
                           model=model,
                           n_eval_episodes=n_eval_episodes,
                           deterministic=deterministic,
                           return_episode_rewards=return_episode_rewards,
                           warn=warn)


def compute_stats(
    experiment: sim.Experiment,
    return_episode_rewards: bool = False
) -> tuple[float, float] | tuple[list[float], list[int]]:
    rs = [
        np.asarray(run.get_record("reward"))
        for run in experiment.runs.values()
    ]
    rewards = np.concatenate([np.sum(r, axis=0) for r in rs])
    lengths = np.concatenate([[r.shape[0]] * r.shape[1] for r in rs])
    if not return_episode_rewards:
        return np.mean(rs), np.std(rs)
    return rewards.tolist(), lengths.tolist()


# returns an array of the sum of agent rewards during runs


def evaluate_with_experiment(
    scenario: sim.Scenario,
    reward: Reward,
    n_eval_episodes: int = 10,
    return_episode_rewards: bool = False,
    time_step: float = 0.1,
    steps: int = 100,
    indices: IndicesLike = Indices.all()
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Evaluate the navground policy/behavior using a navground experiment
    by passing an empty policy to
    :py:func:`navground.learning.evaluation.make_experiment`.

    Arguments ``n_eval_episodes`` and ``return_episode_rewards``
    mimic :py:func:`evaluate` but the return of this function
    is slighty different when ``return_episode_rewards`` is not set:
    it returns the cumulated reward over each episodes for *each agent*,
    not averaging over the group.

    :param      scenario:                The scenario
    :param      reward:                  The reward to record
    :param      n_eval_episodes:         The number of episodes
    :param      return_episode_rewards:  Whether to return all rewards (vs averaging them)
    :param      time_step:               The time step
    :param      steps:                   The steps
    :param      indices:                 The indices of the agents whose
                                         reward we want to record

    :returns:  A tuple of (mean, std dev) or a tuple
       [reward_agent_1_ep_1, reward_agent_2_ep_1, ..., reward_agent_n_ep_m],
       [length_ep_1, length_ep_2, ...]
    """
    group = GroupConfig(reward=reward, indices=Indices(indices))
    exp = make_experiment(scenario=scenario, groups=[group])
    exp.number_of_runs = n_eval_episodes
    exp.time_step = time_step
    exp.steps = steps
    exp.run()
    return compute_stats(exp, return_episode_rewards)


# Mimic SB3 `evaluate_policy`
# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#module-stable_baselines3.common.evaluation


def evaluate_with_experiment_and_env(
    env: BaseEnv | BaseParallelEnv | VecEnv,
    n_eval_episodes: int = 10,
    return_episode_rewards: bool = False,
    indices: IndicesLike = Indices.all()
) -> tuple[list[float], list[int]] | tuple[float, float]:
    """
    Similat to :py:func:`evaluate_with_experiment` but with
    the configuration stored in the environment.

    :param      env:                     The environment
    :param      n_eval_episodes:         The number of episodes
    :param      return_episode_rewards:  Whether to return all rewards (vs averaging them)
    :param      indices:                 The indices of the agents whose
                                         reward we want to record

    :returns:   Same as :py:func:`evaluate_with_experiment`
    """
    group = GroupConfig(indices=Indices(indices))
    exp = make_experiment_with_env(env=env, groups=[group])
    exp.number_of_runs = n_eval_episodes
    exp.run()
    return compute_stats(exp, return_episode_rewards)
