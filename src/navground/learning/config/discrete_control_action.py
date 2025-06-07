from __future__ import annotations

import dataclasses as dc
from collections.abc import Iterable
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core

from ..types import Action
from .control_action import ControlActionConfig


def to_discrete(values: Iterable[float], threshold: float) -> int:
    r = 0
    for i, value in enumerate(values):
        x = 1
        if value > threshold:
            x = 2
        if value < threshold:
            x = 0
        r += (3**i) * x
    return r


def to_continuous(value: int, number: int) -> list[float]:
    rs = []
    for _ in range(number):
        rs.append((value % 3) / 2)
        value = value // 3
    return rs


# TODO(Jerome): handle the case when the continuous low is 0,
# where we need only 2 values instead of 3.

@dc.dataclass
class DiscreteControlActionConfig(ControlActionConfig,
                                  register_name="DiscreteControl"):
    """
    Configuration of the conversion between *discrete* control actions
    and control commands. Actions are either command accelerations
    or command velocities, depending on :py:attr:`ControlActionConfig.use_acceleration_action`.

    Discrete actions are first converted to continuous action and then,
    by the super class :py:class:`ControlActionConfig`, to control commands.

    The action space :py:class:`gymnasium.spaces.Discrete` is discrete.
    Values are encoded using a ternary numeral system,
    where for every dimension:

    .. list-table:: Title
       :header-rows: 1

       * - Discrete
         - Continuous
       * - 0
         - ``space.low``
       * - 1
         - 0
       * - 2
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
        act = np.array(to_continuous(int(action), self.action_size),
                       dtype=self._cont_space.dtype)
        act = self._cont_space.low + act * (self._cont_space.high -
                                            self._cont_space.low)
        return super().get_cmd_from_action(act, behavior, time_step)

    def get_action(self, behavior: core.Behavior, time_step: float) -> Action:
        act = ControlActionConfig.get_action(self, behavior, time_step)
        th = (self._cont_space.high + self._cont_space.low) / 2
        return np.array(to_discrete(act, threshold=th),
                        dtype=np.int64).flatten()

    @property
    def space(self) -> gym.Space[Any]:
        """
        The action space.

        - 0: backwards
        - 1: do-nothing
        - 2: forwards
        """
        return gym.spaces.Discrete(3**self.action_size)
