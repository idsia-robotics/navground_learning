from __future__ import annotations

from navground import core


def compute_speeds(values: list[tuple[float, float]], b) -> list[float]:
    # all have already passed the pad (half-way) or one has already exited the pad
    # => proceed at full speed
    assert len(values) == 2
    ss = [s for _, s in values]
    if all(x > 0 for x, _ in values) or any(x > b for x, _ in values):
        return ss
    # both are in the first half of the pad.
    # if the fastest to exit is moving backwards, let's do it
    if all(x > -b for x, _ in values):
        time_to_exit_fwd = [(b - x) / s for x, s in values]
        time_to_exit_bwd = [(b + x) / s for x, s in values]
        if min(time_to_exit_fwd) > min(time_to_exit_bwd):
            i = 0 if time_to_exit_bwd[0] < time_to_exit_bwd[1] else 1
            ss[i] *= -1
        return ss
    time_to_exit = [(b - x) / s for x, s in values]
    i = 0 if time_to_exit[0] > time_to_exit[1] else 1
    ss[i] = min(ss[i], (-b - values[i][0]) / time_to_exit[1 - i])
    return ss


class PadGroupBehavior(core.BehaviorGroup):

    def __init__(self, pad_width: float = 0.5):
        super().__init__()
        self.pad_width = 0.5

    def compute_cmds(self, time_step: float) -> list[core.Twist2]:
        assert len(self.members) == 2 and all(
            behavior.target.direction is not None for behavior in self.members)
        vs = [
            (
                behavior.target.direction.dot(  # type: ignore[union-attr]
                    behavior.position),
                behavior.get_target_speed()) for behavior in self.members
        ]
        speeds = compute_speeds(vs, self.pad_width / 2)
        return [
            core.Twist2((speed, 0), 0, frame=core.Frame.relative)
            for speed in speeds
        ]


class CentralizedPadBehavior(core.BehaviorGroupMember, name="CentralizedPad"):
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

      where the computation uses the other agent *optimal speed* to predict:

      .. code-block::

         other_agent_time_to_exit_the_pad = (
             other_agent_distance_to_exit_pad / other_agent_optional_speed)``

    """

    _groups: dict[int, core.BehaviorGroup] = {}

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0,
                 pad_width: float = 0.5):
        super().__init__(kinematics, radius)

        self._pad_width = pad_width

    @property
    @core.register(0.5, "Pad width")
    def pad_width(self) -> float:
        return self._pad_width

    @pad_width.setter
    def pad_width(self, value: float) -> None:
        self._pad_width = max(0, value)

    def make_group(self) -> core.BehaviorGroup:
        return PadGroupBehavior(pad_width=self.pad_width)

    def get_groups(self) -> dict[int, core.BehaviorGroup]:
        return self._groups
