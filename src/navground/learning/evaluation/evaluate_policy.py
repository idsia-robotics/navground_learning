from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..env import BaseEnv
from ..types import AnyPolicyPredictor, accept_info

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model: AnyPolicyPredictor,
    env: BaseEnv | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Extends StableBaseline3 :py:func:`stable_baselines3.common.evaluation.evaluate_policy`
    to also accept :py:class:`navground.learning.types.PolicyPredictorWithInfo` models.
    """
    from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv,
                                                  VecMonitor,
                                                  is_vecenv_wrapped)

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([
            lambda: env
        ])

    is_monitor_wrapped = is_vecenv_wrapped(
        env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, "
            "if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
            stacklevel=1,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs
                                      for i in range(n_envs)],
                                     dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    infos = env.reset_infos
    states = None
    episode_starts = np.ones((env.num_envs, ), dtype=bool)
    pass_info = accept_info(model.predict)
    while (episode_counts < episode_count_targets).any():
        if pass_info:
            kwargs = {'info': infos}
        else:
            kwargs = {}
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
            **kwargs)
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():  # noqa
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}")
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
