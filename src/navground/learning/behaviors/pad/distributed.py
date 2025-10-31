from __future__ import annotations

import math

from navground import core
from typing import SupportsFloat


class DistributedPadBehavior(core.Behavior, name="Pad"):
    """
    A centralized behavior that applies a (quasi) optimal strategy
    in :py:class:`navground.learning.scenarios.PadScenario`:

    - if both agents have already passed the mid-point of the pad,
      or at least one agent has already passed the pad:
      they advance at full speed
    - if both agents are in the first half of the pad:
      if the fastest way for one agent to exit first from the path
      is to move backwards:
      the agent move forward at full speed, while the
      other keeps advancing at full speed
    - else:
      the agent predicted to exit the pad second, adjust its speed like

      .. code-block::

         speed = min(optimal_speed,
                     distance_to_enter_pad / other_agent_time_to_exit_the_pad)``

      where the computation uses the other agent *current speed* to predict:

      .. code-block::

         other_agent_time_to_exit_the_pad = (
             other_agent_distance_to_exit_pad / other_agent_current_speed)``
    """

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0,
                 pad_width: float = 0.5):
        """
        Constructs a new instance.

        :param      kinematics:  The kinematics
        :param      radius:      The radius
        :param      pad_width:   The pad width
        """
        super().__init__(kinematics=kinematics, radius=radius)
        self._pad_width = pad_width
        self._state = core.GeometricState()

    @property
    def pad_width(self) -> float:
        """
        The pad width

        :returns:   pad width
        """
        return self._pad_width

    @pad_width.setter
    def pad_width(self, value: float) -> None:
        """
        Sets the pad width

        :param      value:  The desired positive value
        """
        self._pad_width = max(0, value)

    def get_environment_state(self) -> core.EnvironmentState:
        return self._state

    def cmd_twist_towards_velocity(self, velocity: core.Vector2Like,
                                   time_step: SupportsFloat) -> core.Twist2:
        assert len(
            self._state.neighbors) == 1 and self.target.direction is not None
        other = self._state.neighbors[0]
        x = self.target.direction.dot(self.position)
        x_o = -self.target.direction.dot(other.position)
        v_o = abs(self.target.direction.dot(other.velocity))
        b = self.pad_width / 2
        speed = self.optimal_speed
        if x < -b:
            if v_o:
                time_to_enter_o = (-b - x_o) / v_o
            else:
                time_to_enter_o = math.inf
            time_to_enter = (-b - x) / self.optimal_speed
            if x_o > -b or time_to_enter > time_to_enter_o:
                if v_o:
                    time_to_exit_o = (b - x_o) / v_o
                else:
                    time_to_exit_o = math.inf
                if time_to_exit_o > 0:
                    speed = min(self.optimal_speed, (-b - x) / time_to_exit_o)
        elif x < b:
            if abs(x_o) < b:
                # both are inside
                # find minimal time to stop conflict
                if v_o:
                    if x_o > 0:
                        time_to_exit_o = (b - x_o) / v_o
                    else:
                        time_to_exit_o = (b + x_o) / v_o
                else:
                    time_to_exit_o = math.inf
                if x <= 0:
                    time_to_exit = (b + x) / self.optimal_speed
                    if time_to_exit <= time_to_exit_o:
                        speed = -self.optimal_speed
        return core.Twist2((speed, 0), 0, frame=core.Frame.relative)
