from __future__ import annotations

from typing import SupportsFloat

from navground import core


class StopAtPadBehavior(core.Behavior, name="StopAtPad"):
    """
    A centralized behavior for
    :py:class:`navground.learning.scenarios.PadScenario`, that
    stops the agent before entering the pad.
    """

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0,
                 pad_width: float = 0.5,
                 tau: float = 0.5):
        """
        Constructs a new instance.

        :param      kinematics:  The kinematics
        :param      radius:      The radius
        :param      pad_width:   The pad width
        :param      tau:         The time to break
        """
        super().__init__(kinematics=kinematics, radius=radius)
        self._pad_width = pad_width
        self._tau = tau

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

    @property
    def tau(self) -> float:
        """
        The braking time

        :returns:   braking time
        """
        return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        """
        Sets the braking time

        :param      value:  The desired positive value
        """
        self._tau = max(0, value)

    def cmd_twist_towards_velocity(self, velocity: core.Vector2Like,
                                   time_step: SupportsFloat) -> core.Twist2:
        if self.target.direction is None:
            speed = 0.0
        else:
            x = self.target.direction.dot(self.position)
            b = self.pad_width / 2
            if x < -b:
                speed = min(self.optimal_speed, (-b - x) / self.tau)
            else:
                speed = self.optimal_speed
        return core.Twist2((speed, 0), 0, frame=core.Frame.relative)
