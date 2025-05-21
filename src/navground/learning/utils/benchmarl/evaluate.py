from __future__ import annotations

from typing import Any

import numpy as np
from torch import no_grad
from torchrl.envs.utils import (ExplorationType,  # type: ignore
                                set_exploration_type)


def evaluate_policy(
    policy: Any,
    env: Any,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Similar interface as
    StableBaseline3 :py:func:`stable_baselines3.common.evaluation.evaluate_policy`
    but for TorchRL environments (and policies).

    :param      policy:                  The policy
    :param      env:                     The environment
    :param      n_eval_episodes:         The number of episodes
    :param      deterministic:           Whether to evaluate the policy deterministic
    :param      return_episode_rewards:  Whether to return individual episode rewards
                                         (vs aggregate them)
    :param      warn:                    Whether to emit warnings

    :returns:   If ``return_episode_rewards`` is set,
        a tuple (list of cumulated episodes rewards', list of episodes length)
        else, a tuple (average episodes rewards, std dev of episodes rewards)
    """
    num = len(env.env.possible_agents)
    episode_rewards = []
    episode_lengths = []
    for _ in range(n_eval_episodes // num):
        with set_exploration_type(ExplorationType.DETERMINISTIC), no_grad():
            rollout = env.rollout(max_steps=10_000, policy=policy)
        reward = rollout['next']['agent']['reward'].numpy()
        episode_rewards.append(np.sum(reward, axis=0).flatten())
        length = len(reward)
        episode_lengths.append(length)
        episode_lengths.append(length)
    episode_rewards = np.concatenate(episode_rewards)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
