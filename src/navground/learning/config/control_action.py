from __future__ import annotations

import dataclasses as dc
import math
import warnings

import gymnasium as gym
import numpy as np
from navground import core
from navground.core import FloatType

from ..types import Array
from .base import ActionConfig, ConfigWithKinematic


@dc.dataclass(repr=False)
class ControlActionConfig(ConfigWithKinematic,
                          ActionConfig,
                          register_name="Control"):
    """
    Configuration of the conversion between control actions
    and control commands. Actions are either command accelerations
    or command velocities, depending on :py:attr:`use_acceleration_action`.

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

    """
    max_acceleration: float = np.inf
    """The maximal agent acceleration module; only relevant
    if :py:attr:`use_acceleration_action` is set.
    """
    max_angular_acceleration: float = np.inf
    """The maximal agent angular acceleration; only relevant
    if :py:attr:`use_acceleration_action` is set.
    """
    use_acceleration_action: bool = False
    """Whether to output acceleration commands instead of
    velocity commands.
    """
    fix_orientation: bool = False
    """Whether to keep the agent orientation fixed"""
    use_wheels: bool = False
    """Whether to output control commands that refer to wheel speeds
    or accelerations."""
    has_wheels: bool | None = None
    """Whether the agent as wheels. If None, it will defer to the agent kinematics."""

    @property
    def should_use_wheels(self) -> bool:
        if self.use_wheels:
            if self.has_wheels is None:
                warnings.warn("Set has_wheels first", stacklevel=1)
            return bool(self.has_wheels)
        return False

    def is_configured(self, warn: bool = False) -> bool:
        if self.use_wheels and self.has_wheels is None:
            if warn:
                warnings.warn(
                    "Configured to use wheels but does not know whether the agent has wheels",
                    stacklevel=1)

            return False
        if self.fix_orientation and self.dof is None:
            if warn:
                warnings.warn(
                    "Configured to keep orientation fixed but does not know the number of dof",
                    stacklevel=1)

            return False
        if self.use_acceleration_action:
            if not math.isfinite(self.max_acceleration) or self.dof is None:
                if warn:
                    warnings.warn(
                        "Configured to output accelerations but does not know max acceleration",
                        stacklevel=1)

                return False
            if not self.should_fix_orientation and not self.should_use_wheels and not math.isfinite(
                    self.max_angular_acceleration):
                if warn:
                    warnings.warn(
                        "Configured to output accelerations but "
                        "does not know max angular acceleration",
                        stacklevel=1)
                return False
        else:
            if not math.isfinite(self.max_speed):
                if warn:
                    warnings.warn(
                        "Configured to output velocities but does not know max speed",
                        stacklevel=1)
                return False
            if not self.should_fix_orientation and not self.should_use_wheels and not math.isfinite(
                    self.max_angular_speed):
                if warn:
                    warnings.warn(
                        "Configured to output velocities but does not know max angular speed",
                        stacklevel=1)
                return False
        return True

    def configure(self, behavior: core.Behavior) -> None:
        super().configure_kinematics(behavior)
        if self.has_wheels is None:
            if behavior.kinematics:
                self.has_wheels = behavior.kinematics.is_wheeled()
            else:
                self.has_wheels = False

    @property
    def should_fix_orientation(self) -> bool:
        if self.dof is None:
            raise ValueError("Set the DOF first")
        return self.fix_orientation and self.dof > 2

    @property
    def action_size(self) -> int:
        if self.should_use_wheels:
            if self.dof == 2:
                return 2
            else:
                return 4
        if self.dof == 2 or self.fix_orientation:
            return 2
        return 3

    @property
    def space(self) -> gym.spaces.Box:
        """
        The action space.
        """
        return gym.spaces.Box(-1,
                              1,
                              shape=(self.action_size, ),
                              dtype=self.box_type)

    def _compute_wheels_value_from_action(self, action: Array,
                                          max_value: float) -> Array:
        return action * max_value

    def _compute_value_from_action(self, action: Array, max_value: float,
                                   max_angular_value: float) -> core.Twist2:
        if self.dof == 2 or not self.should_fix_orientation:
            angular_value = action[-1] * max_angular_value
        else:
            angular_value = 0
        if self.dof == 2:
            value = np.array((action[0] * max_value, 0), dtype=FloatType)
        else:
            value = np.asarray(action[:2], dtype=FloatType) * max_value
        twist = core.Twist2(velocity=value,
                            angular_speed=angular_value,
                            frame=core.Frame.relative)
        return twist

    # DONE(Jerome): we were not rescaling the speed
    def get_cmd_from_action(self, action: Array,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        if self.should_use_wheels:
            assert behavior is not None
            if self.use_acceleration_action:
                accs = self._compute_wheels_value_from_action(
                    action, self.max_acceleration)
                speeds = np.asarray(behavior.wheel_speeds) + accs * time_step
                return behavior.twist_from_wheel_speeds(speeds)
            speeds = self._compute_wheels_value_from_action(
                action, self.max_speed)
            return behavior.twist_from_wheel_speeds(speeds)
        if self.use_acceleration_action:
            assert behavior is not None
            acc = self._compute_value_from_action(
                action, self.max_acceleration, self.max_angular_acceleration)
            twist = behavior.get_twist(frame=core.Frame.relative)
            return core.Twist2(twist.velocity + time_step * acc.velocity,
                               twist.angular_speed +
                               time_step * acc.angular_speed,
                               frame=core.Frame.relative)
        return self._compute_value_from_action(action, self.max_speed,
                                               self.max_angular_speed)

    def _action_from_twist(self, value: core.Twist2, max_value: float,
                           max_angular_value: float) -> Array:
        if max_value:
            v = value.velocity[:2] / max_value
        else:
            v = np.zeros(2, dtype=FloatType)
        if max_angular_value:
            w = value.angular_speed / max_angular_value
        else:
            w = 0
        if self.dof == 2:
            return np.array([v[0], w], FloatType)
        elif self.should_fix_orientation:
            return v
        return np.array([*v, w], FloatType)

    def _action_from_wheels(self, values: Array, max_value: float) -> Array:
        if max_value:
            return values / max_value
        return np.zeros(self.action_size, dtype=FloatType)

    def _compute_action_from_wheel_speeds(self,
                                          behavior: core.Behavior) -> Array:
        ws = np.array(behavior.actuated_wheel_speeds, dtype=FloatType)
        return self._action_from_wheels(ws, self.max_speed)

    def _compute_action_from_wheel_accelerations(self, behavior: core.Behavior,
                                                 time_step: float) -> Array:
        ws = np.array(behavior.actuated_wheel_speeds, dtype=FloatType)
        cs = np.array(behavior.wheel_speeds, dtype=FloatType)
        return self._action_from_wheels((ws - cs) / time_step,
                                        self.max_acceleration)

    def _compute_action_from_acceleration(self, behavior: core.Behavior,
                                          time_step: float) -> Array:
        cmd = behavior.get_actuated_twist(core.Frame.relative)
        twist = behavior.get_twist(core.Frame.relative)
        acc = core.Twist2(
            (cmd.velocity - twist.velocity) / time_step,
            (cmd.angular_speed - twist.angular_speed) / time_step)
        return self._action_from_twist(acc, self.max_acceleration,
                                       self.max_angular_acceleration)

    def _compute_action_from_velocity(self, behavior: core.Behavior) -> Array:
        cmd = behavior.get_actuated_twist(core.Frame.relative)
        return self._action_from_twist(cmd, self.max_speed,
                                       self.max_angular_speed)

    def get_action(self, behavior: core.Behavior, time_step: float) -> Array:
        if self.should_use_wheels:
            if self.use_acceleration_action:
                acts = self._compute_action_from_wheel_accelerations(
                    behavior, time_step)
            else:
                acts = self._compute_action_from_wheel_speeds(behavior)
        elif self.use_acceleration_action:
            acts = self._compute_action_from_acceleration(behavior, time_step)
        else:
            acts = self._compute_action_from_velocity(behavior)
        return np.clip(acts, -1, 1)
