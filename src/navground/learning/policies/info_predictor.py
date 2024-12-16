from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..types import Action, EpisodeStart, Info, Observation, State


class InfoPolicy:
    """
    A predictor that extracts navground actions
    from the info dictionary and conforms to
    :py:class:`navground.learning.types.PolicyPredictorWithInfo`

    :param action_space:       The action space
    :param key:                The key of the action in the ``info`` dictionary
    :param observation_space:  An optional observation space
    """

    def __init__(self,
                 action_space: gym.Space[Any],
                 key: str,
                 observation_space: gym.Space[Any] = gym.spaces.Dict()) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.key = key
        self._default = self.action_space.sample() * 0

    def get_actions(self, info: Info) -> Action:
        if isinstance(info, list):
            return np.asarray([i.get(self.key, self._default) for i in info])
        return info.get(self.key, self._default)

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False,
                info: Info | None = None) -> tuple[Action, State | None]:
        assert info is not None
        return self.get_actions(info), None

    def __call__(self,
                 observation: Observation,
                 state: State | None = None,
                 episode_start: EpisodeStart | None = None,
                 info: Info | None = None) -> tuple[Action, State | None]:
        return self.predict(observation, state, episode_start, True, info)
