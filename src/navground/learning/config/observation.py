from __future__ import annotations

import dataclasses as dc
import math
import warnings
from collections.abc import Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core
from navground.core import FloatType

from ..types import Array, Observation
from .base import ConfigWithKinematic, ObservationConfig


@dc.dataclass(repr=False)
class DefaultObservationConfig(ConfigWithKinematic,
                               ObservationConfig,
                               register_name="Default"):
    """
    This class configures which information from the behavior state to
    include in the observations.

    :param flat: Whether to flatten the observation space

    :param history: The size of observations queue.
       If larger than 1, recent observations will be first stacked and then flattened.

    :param include_target_distance: Whether to include the target distance in the observations.

    :param include_target_distance_validity: Whether to include whether the target
                                             distance is valid in the observations.

    :param max_target_distance: The upper bound of target distance.
                                Only relevant if ``include_target_distance=True``

    :param include_target_direction: Whether to include the target direction in the observations.

    :param include_target_direction_validity: Whether to include whether the target direction
                                              is valid in the observations.

    :param include_velocity: Whether to include the current velocity in the observations.

    :param include_angular_speed: Whether to include the current angular_speed in the observations.

    :param include_target_speed: Whether to include the target speed in the observations.

    :param include_target_angular_speed: Whether to include the target angular speed
                                         in the observations.

    :param max_speed: The upper bound of the speed.

    :param max_angular_speed: The upper bound of the angular speed.

    :param include_radius: Whether to include the own radius in the observations.

    :param max_radius: The upper bound of own radius.
                       Only relevant if ``include_radius=True``.
    """

    flat: bool = False
    """Whether the observation space is flat"""
    history: int = 1
    """The size of observations queue.
       If larger than 1, recent observations will be first stacked and then flattened."""
    include_target_distance: bool = False
    """Whether observations include the target direction."""
    include_target_distance_validity: bool = False
    """Whether observations include the validity of the target direction."""
    max_target_distance: float = np.inf
    """The upper bound of target distance.
       Only relevant if :py:attr:`include_target_distance` is set."""
    include_target_direction: bool = True
    """Whether observations include the target direction."""
    include_target_direction_validity: bool = False
    """Whether observations include whether the validity of the target direction."""
    include_velocity: bool = False
    """Whether observations include the current velocity."""
    include_angular_speed: bool = False
    """Whether observations include the current angular_speed."""
    include_radius: bool = False
    """Whether observations include the own radius."""
    include_target_speed: bool = False
    """Whether observations include the target speed."""
    include_target_angular_speed: bool = False
    """Whether observations include the target angular speed."""
    max_radius: float = np.inf
    """The upper bound of own radius.
       Only relevant if :py:attr:`include_radius` is set."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._item_space: gym.spaces.Dict = gym.spaces.Dict()

    def is_configured(self, warn: bool = False) -> bool:
        if not self._item_space:
            if warn:
                warnings.warn("Observation space is empty", stacklevel=1)
            return False
        if not math.isfinite(self.max_speed):
            if self.include_velocity or self.include_target_speed:
                if warn:
                    warnings.warn("Does not know max ego speed", stacklevel=1)
                return False
        if self.dof is None and self.include_velocity:
            if warn:
                warnings.warn("Does not know the number of dof", stacklevel=1)
            return False
        if not math.isfinite(self.max_angular_speed):
            if self.include_angular_speed or self.include_target_angular_speed:
                if warn:
                    warnings.warn("Does not know max ego angular speed",
                                  stacklevel=1)
                return False
        if not math.isfinite(
                self.max_target_distance) and self.include_target_distance:
            if warn:
                warnings.warn("Does not know max target distance",
                              stacklevel=1)
            return False
        if not math.isfinite(self.max_radius) and self.include_radius:
            if warn:
                warnings.warn("Does not know max ego radius", stacklevel=1)
            return False
        return True

    def configure(self, behavior: core.Behavior | None,
                  sensing_space: gym.spaces.Dict) -> None:
        if behavior:
            super().configure_kinematics(behavior)
            if not math.isfinite(self.max_target_distance):
                self.max_target_distance = behavior.horizon
            if not math.isfinite(self.max_radius):
                self.max_radius = behavior.radius
        self._item_space = self._make_item_space(sensing_space)

    def _make_state_space(self) -> gym.spaces.Dict:
        ds: dict[str, gym.Space[Any]] = {}
        if self.include_target_direction:
            ds['ego_target_direction'] = gym.spaces.Box(-1,
                                                        1, (2, ),
                                                        dtype=FloatType)
            if self.include_target_direction_validity:
                ds['ego_target_direction_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
                # gym.spaces.Discrete(n=2)
        if self.include_target_distance:
            ds['ego_target_distance'] = gym.spaces.Box(
                0, self.max_target_distance, (1, ), dtype=FloatType)
            if self.include_target_direction_validity:
                ds['ego_target_distance_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
        if self.include_velocity:
            if self.dof is None:
                raise ValueError("Set the DOF first")
            ds['ego_velocity'] = gym.spaces.Box(-self.max_speed,
                                                self.max_speed,
                                                (self.dof - 1, ),
                                                dtype=FloatType)
        if self.include_angular_speed:
            ds['ego_angular_speed'] = gym.spaces.Box(-self.max_angular_speed,
                                                     self.max_angular_speed,
                                                     (1, ),
                                                     dtype=FloatType)
        if self.include_radius:
            ds['ego_radius'] = gym.spaces.Box(0,
                                              self.max_radius, (1, ),
                                              dtype=FloatType)
        if self.include_target_speed:
            ds['ego_target_speed'] = gym.spaces.Box(0,
                                                    self.max_speed, (1, ),
                                                    dtype=FloatType)
        if self.include_target_angular_speed:
            ds['ego_target_angular_speed'] = gym.spaces.Box(
                0, self.max_angular_speed, (1, ), dtype=FloatType)
        return gym.spaces.Dict(ds)

    def get_observation(self, behavior: core.Behavior | None,
                        buffers: Mapping[str, core.Buffer]) -> Observation:
        rs = self._get_state_observations(behavior)
        rs.update((k, b.data) for k, b in buffers.items())
        if not self.should_flatten_observations:
            return rs
        vs = cast(Array, gym.spaces.flatten(self._item_space, rs))
        if self._dtype:
            vs = vs.astype(self._dtype)
        return vs

    @property
    def should_flatten_observations(self) -> bool:
        return self.flat or self.history > 1

    def _make_item_space(self,
                         sensing_space: gym.spaces.Dict) -> gym.spaces.Dict:
        return gym.spaces.Dict(**sensing_space, **self._make_state_space())

    @property
    def space(self) -> gym.Space[Any]:
        if self.should_flatten_observations:
            flat_space: gym.spaces.Box = cast(
                gym.spaces.Box, gym.spaces.flatten_space(self._item_space))
            if self._dtype:
                flat_space.dtype = self._dtype
            if self.history > 1:
                low = np.repeat(flat_space.low[np.newaxis, ...],
                                self.history,
                                axis=0)
                high = np.repeat(flat_space.high[np.newaxis, ...],
                                 self.history,
                                 axis=0)
                flat_space = gym.spaces.Box(low=low,
                                            high=high,
                                            dtype=self.box_type)
            return flat_space
        return self._item_space

    def _get_state_observations(
        self,
        behavior: core.Behavior | None,
    ) -> dict[str, Array]:
        rs: dict[str, Array] = {}
        if behavior:
            if self.include_velocity:
                v = core.to_relative(behavior.velocity, behavior.pose)
                if self.dof == 2:
                    rs['ego_velocity'] = v[:1]
                else:
                    rs['ego_velocity'] = v
            if self.include_angular_speed:
                rs['ego_angular_speed'] = np.array([behavior.angular_speed],
                                                   dtype=FloatType)
            if self.include_radius:
                rs['ego_radius'] = np.array([behavior.radius], dtype=FloatType)
            self._add_target(behavior, rs)
        return rs

    def _add_target(self, behavior: core.Behavior, rs: dict[str,
                                                            Array]) -> None:
        if self.include_target_distance:
            distance = behavior.get_target_distance()
            rs['ego_target_distance'] = np.array(
                [min(distance or 0.0, self.max_target_distance)],
                dtype=FloatType)
            if self.include_target_distance_validity:
                value = 0 if distance is None else 1
                rs['ego_target_distance_valid'] = np.array([value], np.uint8)
        if self.include_target_direction:
            e = behavior.get_target_direction(core.Frame.relative)
            rs['ego_target_direction'] = e if e is not None else np.zeros(
                2, dtype=FloatType)
            if self.include_target_direction_validity:
                value = 0 if e is None else 1
                rs['ego_target_direction_valid'] = np.array([value], np.uint8)
        if self.include_target_speed:
            rs['ego_target_speed'] = np.array(
                [min(behavior.get_target_speed(), self.max_speed)],
                dtype=FloatType)

        if self.include_target_angular_speed:
            w = min(behavior.get_target_angular_speed(),
                    self.max_angular_speed)
            rs['ego_target_angular_speed'] = np.array([w], dtype=FloatType)

    # def _add_target(self, behavior: core.Behavior,
    #                 rs: dict[str, np.ndarray]) -> None:
    #     if behavior.target.position is not None:
    #         p = core.to_relative(behavior.target.position - behavior.position,
    #                              behavior.pose)
    #         dist = np.linalg.norm(p)
    #         if self.include_target_direction:
    #             if dist > 0:
    #                 p = p / dist
    #             rs['ego_target_direction'] = p
    #         if self.include_target_distance:
    #             rs['ego_target_distance'] = np.array(
    #                 [min(dist, behavior.horizon)])
    #     elif behavior.target.direction is not None and self.include_target_direction:
    #         rs['ego_target_direction'] = core.to_relative(
    #             behavior.target.direction, behavior.pose)
