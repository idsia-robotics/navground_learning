from __future__ import annotations

import dataclasses as dc
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core

from ..types import Action
from .control_action import ControlActionConfig
from .discrete_control_action import to_continuous, to_discrete


def to_multibinary(value: int, number: int) -> list[int]:
    rs = []
    for _ in range(number):
        rs.append(value % 2)
        value = value // 2
    return rs


@dc.dataclass
class DiscreteControlActionWithCommConfig(
        ControlActionConfig, register_name="DiscreteControlWithComm"):
    """
    Actions are composed of control values and the message to be transmitted

    Message {0, 1} and controls {0, 1, 2} are discrete and
    encoded using a mixed numerical scheme:
    - the lower part with a ternary encoding for controls
    - the upper part with binary encoding for messages.

    Action to be actuated are decomposed:
    - the control is converted to a :py:class:`navground.core.Twist2`,
    - the message is set as attribute ``Behavior._comm``.

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

    :param comm_size: The size of the message to broadcast

    """
    comm_size: int = 1
    """The size of the message to broadcast"""

    @property
    def comm_space(self) -> gym.spaces.Box:
        """
        The box space of the broadcasted messages with dimension
        :py:meth:`ControlActionConfig.action_size` and values in {0, 1}.

        :returns: the box space
        """
        return gym.spaces.Box(low=0,
                              high=1,
                              shape=(self.comm_size, ),
                              dtype=np.int8)

    def configure(self, behavior: core.Behavior) -> None:
        super().configure(behavior)
        self._cont_space = cast('gym.spaces.Box', super().space)

    def get_cmd_from_action(self, action: Action,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        n = 3**self.action_size
        comm = to_multibinary(int(action) // n, self.comm_size)
        if behavior:
            behavior._comm = np.array(comm, dtype=np.int8)  # type: ignore[attr-defined]
        action = int(action) % n
        act = np.array(to_continuous(int(action), self.action_size),
                       dtype=self._cont_space.dtype)
        act = self._cont_space.low + act * (self._cont_space.high -
                                            self._cont_space.low)
        return super().get_cmd_from_action(act, behavior, time_step)

    def get_action(self, behavior: core.Behavior, time_step: float) -> Action:
        act = ControlActionConfig.get_action(self, behavior, time_step)
        if hasattr(behavior, '_comm'):
            comm = behavior._comm
        else:
            comm = []
        if len(comm) < self.comm_size:
            comm = comm + [0] * (self.comm_size - len(comm))
        th = (self._cont_space.high + self._cont_space.low) / 2
        dact = to_discrete(act, threshold=th)
        dcomm = sum(v * (2**i) for i, v in enumerate(comm))
        d = dact + dcomm * (3**self.action_size)
        return np.array([d], dtype=np.int64)

    @property
    def space(self) -> gym.Space[Any]:
        """
        The control space {0 (backwards), 1 (do-nothing), 2 (forwards)}
        multiplied by the communication space {0, 1}
        """
        return gym.spaces.Discrete(3**self.action_size * 2**self.comm_size)
