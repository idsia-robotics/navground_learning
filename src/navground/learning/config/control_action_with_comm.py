from __future__ import annotations

import dataclasses as dc

import gymnasium as gym
import numpy as np
from navground import core

from ..types import Action
from .control_action import ControlActionConfig


@dc.dataclass
class ControlActionWithCommConfig(ControlActionConfig,
                                  register_name="ControlWithComm"):
    """
    Configuration of the conversion between actions
    and control commands.

    Actions are composed of control values (accelerations
    or command velocities, depending on
    :py:meth:`ControlActionConfig.use_acceleration_action`)
    and the message to be transmitted.

    Action are decomposed as:

    - the control is converted to a :py:class:`navground.core.Twist2` (and later actuated)
    - the message is set as attribute ``Behavior._comm`` (and later broadcasted)

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
    comm_size: int = 2
    """The size of the message to broadcast"""

    @property
    def action_size(self) -> int:
        return super().action_size + self.comm_size

    @property
    def comm_space(self) -> gym.spaces.Box:
        """
        The box space of the broadcasted messages with dimension
        :py:meth:`ControlActionConfig.action_size`
        and values between -1 and 1.

        :returns: the box space
        """
        return gym.spaces.Box(low=-1,
                              high=1,
                              shape=(self.comm_size, ),
                              dtype=core.FloatType)

    def get_cmd_from_action(self, action: Action,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        cmd = super().get_cmd_from_action(action, behavior, time_step)
        if behavior:
            behavior._comm = action[  # type: ignore[attr-defined]
                -self.comm_size:]
        return cmd

    def get_action(self, behavior: core.Behavior, time_step: float) -> Action:
        act = ControlActionConfig.get_action(self, behavior, time_step)
        if hasattr(behavior, '_comm'):
            comm = behavior._comm  # type: ignore[attr-defined]
        else:
            comm = np.zeros(self.comm_size, dtype=core.FloatType)
        act = np.concatenate([act, comm], dtype=core.FloatType)
        return act
