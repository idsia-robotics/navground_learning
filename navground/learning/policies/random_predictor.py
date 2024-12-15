from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import get_number_of_batches

if TYPE_CHECKING:
    import gymnasium as gym

    from ..types import Action, EpisodeStart, Observation, State


class RandomPredictor:
    """
    This class describes a predictor that returns random actions and
    conforms to :py:class:`navground.learning.types.PolicyPredictor`

    :param action_space:       The action space
    :param observation_space:  An optional observation space
    """
    def __init__(
        self,
        action_space: gym.spaces.Box,
        observation_space: gym.Space[Any],
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def get_actions(
        self,
        obs: Observation,
    ) -> Action:
        number = get_number_of_batches(obs, self.observation_space)
        if number == 0:
            return self.action_space.sample()
        return np.stack([self.action_space.sample() for _ in range(number)],
                        axis=0)

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False) -> tuple[Action, State | None]:
        return self.get_actions(observation), None

    def __call__(
        self,
        observation: Observation,
        state: State | None = None,
        episode_start: EpisodeStart | None = None
    ) -> tuple[Action, State | None]:
        return self.predict(observation, state, episode_start, True)
