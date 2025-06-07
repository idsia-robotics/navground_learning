from __future__ import annotations

import dataclasses as dc
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core

from ..types import Action
from .control_action import ControlActionConfig


def to_discrete(value: float, threshold: float = 0) -> tuple[int, int]:
    if value > threshold:
        return (0, 1)
    if value < threshold:
        return (1, 0)
    return (0, 0)


def to_continuous(values: tuple[int, int]) -> float:
    if values == (1, 0):
        return 0
    if values == (0, 1):
        return 1
    return 0.5


@dc.dataclass
class BinaryControlActionConfig(ControlActionConfig,
                                register_name="BinaryControl"):
    """
    Configuration of the conversion between *discrete* control actions
    and control commands. Actions are either command accelerations
    or command velocities, depending on :py:attr:`ControlActionConfig.use_acceleration_action`.

    Discrete actions are first converted to continuous action and then,
    by the super class :py:class:`ControlActionConfig`, to control commands.

    The action space :py:class:`gymnasium.spaces.MultiBinary` is discrete,
    with two values for each continuous dimension:
    the first bit refers to moving/accelerating backwards (yes/no),
    and the second bit to moving/accelerating forwards (yes/no).

    Discrete values are encoded using a 2-bits per dimension, where

    .. list-table:: Title
       :header-rows: 1

       * - Discrete
         - Continuous
       * - [1, 0]
         - ``space.low``
       * - [0, 0]
         - 0
       * - [1, 1]
         - 0
       * - [0, 1]
         - ``space.high``

    :param dtype: The data type

    :param dof: The number of degrees of freedom of the agent

    :param max_speed: The upper bound of the speed.

    :param max_angular_speed: The upper bound of the angular speed.

    :param max_acceleration: The upper bound of the acceleration.

    :param max_angular_acceleration: The upper bound of the angular acceleration.

    :param use_acceleration_action: Whether actions are accelerations.
                                    If not set, actions are velocities.

    :param use_wheels: Whether action uses wheel speeds/acceleration
                       instead of body speeds/acceleration.
                       Only effective if the b behavior has a wheeled kinematics.

    :param fix_orientation: Whether to force the agent not to control orientation,
                            i.e., to not include the angular command in actions.

    :param has_wheels: Whether the agent as wheels.
                       If None, it will defer to the agent kinematics.
    """

    def configure(self, behavior: core.Behavior) -> None:
        super().configure(behavior)
        self._cont_space = cast('gym.spaces.Box', super().space)

    def get_cmd_from_action(self, action: Action,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        low = self._cont_space.low
        high = self._cont_space.high
        act = action.reshape((-1, 2)).astype(self._cont_space.dtype)
        act = ((act[..., 1] - act[..., 0]) / 2 + 0.5) * (high - low) + low
        return super().get_cmd_from_action(act, behavior, time_step)

    def get_action(self, behavior: core.Behavior, time_step: float) -> Action:
        act = ControlActionConfig.get_action(self, behavior, time_step)
        ths = (self._cont_space.high + self._cont_space.low) / 2
        d_act = np.array(
            [to_discrete(x, th) for x, th in zip(act, ths, strict=True)],
            dtype=np.int8).flatten()
        return d_act

    @property
    def space(self) -> gym.Space[Any]:
        """
        The action space.

        brake Y/N x accelerate Y/N
        """
        return gym.spaces.MultiBinary(self.action_size * 2)
