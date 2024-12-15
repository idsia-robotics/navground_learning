from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np

from ..indices import Indices, IndicesLike
from ..parallel_env import BaseParallelEnv
from ..types import (Array, Observation, AnyPolicyPredictor, State,
                     accept_info)


def stack_dict(values: dict[Any, Array]) -> Array:
    return np.stack(list(values.values()))


def stack_obs_dict(values: dict[Any, dict[str, Array]],
                   keys: Iterable[str]) -> dict[str, Array]:
    return {
        k: stack_dict({
            i: vs[k]
            for i, vs in values.items()
        })
        for k in keys
    }


def evaluate_policies(
    models: Sequence[tuple[IndicesLike, AnyPolicyPredictor]],
    env: BaseParallelEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[list[float], list[float]] | tuple[list[list[float]], list[int],
                                             list[list[int]]]:
    """
    Mimics StableBaseline3 :py:func:`stable_baselines3.common.evaluation.evaluate_policy`
    on a :py:class:`PettingZoo  Parallel environment <pettingzoo.utils.env.ParallelEnv>`.

    The main differences are:

    - it accepts a list of models to be applied to different sub-groups of agents
    - it returns the rewards divided by groups.

    For example, if a single group is provided

    >>> evaluate_policies(models=[(Indices.all(), model)], env=env)
    ([100.0], [100.0])

    it returns a list of with a single value for the average
    and standard deviation over all episodes of the mean reward over all agents.

    Instead, if we pass two groups, like

    >>> models=[({1, 2}, model1), ({3, 4}, model2), ({5, 6}, model3)]
    >>> evaluate_policies(models=models, env=env)
    ([100.0, 90.0, 110.0], [10.0, 12.0, 7.0])

    it returns two lists of three elements each, one for each group.

    If `return_episode_rewards` is set, it returns three lists:

    - the cumulated rewards for each group and episodes
      (not averaged over the agents!),
      ``[[grp_1_ep_1, grp_1_ep_2, ...], [grp_2_ep_1, grp_2_ep_2, ...]``
    - the length of the episodes
    - the number of agents for each group and episodes.
      ``[[grp_1_ep_1, grp_1_ep_2, ...], [grp_2_ep_1, grp_2_ep_2, ...]``

    For example, with two groups and three episodes, it will be like

    >>> models=[({1, 2}, model1), ({3, 4, 5}, model2)]
    >>> evaluate_policies(models=models, env=env,
                          n_eval_episodes=2,
                          return_episode_rewards=False)
    ([[200.0, 202.0], [300.0, 305.0]], [10, 10], [[2, 2], [3, 3]])


    :param      models:    The models as tuples of indices
                           selecting the agents in the group
                           and the model they will apply

    :param      env:                     The environment

    :param      n_eval_episodes:         The number of episodes

    :param      deterministic:           Whether the policy is applied deterministically

    :param      return_episode_rewards:  Whether to return all episodes (vs averaging them)

    :param      warn:                    Whether to enable warnings

    :returns:   If ``return_episode_rewards`` is set,
        a tuple (list of lists of cumulated episodes rewards',
        list of episodes length,
        list of lists of size of groups),
        else, a tuple (list of average episodes rewards,
        list of std dev of episodes rewards)
    """
    model_indices = [Indices(indices) for indices, _ in models]
    # The cumulated reward of agents in a group
    episode_rewards: list[list[float]] = [[] for _ in models]
    episode_lengths: list[int] = []
    # The number of agents in a group
    episode_numbers: list[list[int]] = [[] for _ in models]
    model_use_info = [accept_info(model.predict) for _, model in models]
    for i in range(n_eval_episodes):
        # The cumulated reward of agents in a group
        observations, infos = env.reset()
        groups = [
            indices.sub_sequence(env.agents) for indices in model_indices
        ]
        # print(env.agents, groups, models)
        current_numbers = [len(group) for group in groups]
        current_reward: list[float] = [0 for group in groups]
        current_length = 0
        group_state: list[State | None] = [None for i in groups]
        done = False
        dones: Array = np.ones((len(env.agents), ), dtype=np.bool)
        obs_spaces = [
            env.observation_space(group[0]) if group else None
            for group in groups
        ]
        group_obs_keys = [
            space.keys() if isinstance(space, gym.spaces.Dict) else None
            for space in obs_spaces
        ]
        while not done:
            actions = {}
            for i, (group, indices, (_, model), use_info,
                    obs_keys) in enumerate(
                        zip(groups,
                            model_indices,
                            models,
                            model_use_info,
                            group_obs_keys,
                            strict=False)):
                if not group:
                    continue
                group_obs = indices.sub_dict(observations)
                if obs_keys:
                    obs: Observation = stack_obs_dict(
                        cast(dict[Any, dict[str, Array]], group_obs), obs_keys)
                else:
                    obs = stack_dict(cast(dict[int, Array], group_obs))
                episode_start = dones[group]
                if use_info:
                    kwargs = {'info': list(indices.sub_dict(infos).values())}
                else:
                    kwargs = {}
                acts, group_state[i] = model.predict(
                    obs,
                    state=group_state[i],
                    episode_start=episode_start,
                    deterministic=deterministic,
                    **kwargs)
                for i, action in zip(group, acts, strict=False):
                    actions[i] = action
            observations, rewards, terminated, truncated, infos = env.step(
                actions)

            for i, indices in enumerate(model_indices):
                current_reward[i] += float(
                    sum(indices.sub_dict(rewards).values()))
            dones = np.bitwise_or(list(terminated.values()),
                                  list(truncated.values()))
            done = bool(np.all(dones))
            current_length += 1

        episode_lengths.append(current_length)
        for i, number in enumerate(current_numbers):
            episode_numbers[i].append(number)
        for i, reward in enumerate(current_reward):
            episode_rewards[i].append(reward)
    if not return_episode_rewards:
        rews = [
            np.asarray(rs) / np.asarray(ns)
            for rs, ns in zip(episode_rewards, episode_numbers, strict=False)
        ]
        mean_reward = [np.mean(rew) for rew in rews]
        std_reward = [np.std(rew) for rew in rews]
        return mean_reward, std_reward
    return episode_rewards, episode_lengths, episode_numbers
