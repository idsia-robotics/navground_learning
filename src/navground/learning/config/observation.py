from __future__ import annotations

import dataclasses as dc
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, TypeAlias, cast

import gymnasium as gym
import numpy as np
from navground import core
from navground.core import FloatType

from ..types import Array, Observation
from .base import ConfigWithKinematic, ObservationConfig

RescaleFn: TypeAlias = Callable[[Array], Array]


def flatten_dict_space(space: gym.spaces.Dict) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        k: gym.spaces.flatten_space(v)
        for k, v in space.items()
    })


def normalize(low: Array, high: Array, new_low: float,
              new_high: float) -> RescaleFn:

    scale = (new_high - new_low) / (high - low)

    def f(x: Array) -> Array:
        return (x - low) * scale + new_low

    return f


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

    :param normalize: Whether to normalize observations in [-1, 1]

    :param ignore_keys: Which keys to ignore

    :param sort_keys: Whether to sort the keys

    :param keys: Which keys to select

    :param include_position: Whether to include the behavior position
        Coordinates are only included if the related interval has length > 0,
        e.g., ``min_x < max`` for the x-coordinate.

    :param max_x: The upper bound of x-coordinate

    :param max_y: The upper bound of y-coordinate

    :param min_x: The lower bound of x-coordinate

    :param min_y: The lower bound of y-coordinate

    :param include_orientation: Whether to include the behavior orientation

    """

    flat: bool = False
    """Whether the observation space is flat"""
    flat_values: bool = False
    """Whether the flatten the spaces of a dict observation space"""
    history: int = 1
    """The size of observations queue.
       If larger than 1, recent observations will be first stacked and then flattened."""
    include_target_orientation: bool = False
    """TODO"""
    include_target_orientation_validity: bool = False
    """TODO"""
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
    normalize: bool = False
    """Whether to normalize observations in [-1, 1]"""
    ignore_keys: list[str] = dc.field(default_factory=list)
    """Which keys to ignore"""
    sort_keys: bool = False
    """Whether to sort the keys"""
    keys: Sequence[str] | None = None
    """Which keys to select"""
    include_position: bool = False
    """Whether to include the behavior position"""
    max_x: float = math.inf
    """The maximal x-coordinate"""
    max_y: float = math.inf
    """The maximal y-coordinate"""
    min_x: float = -math.inf
    """The minimal x-coordinate"""
    min_y: float = -math.inf
    """The minimal y-coordinate"""
    include_orientation: bool = False
    """Whether to include the behavior orientation"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._item_space: gym.spaces.Dict = gym.spaces.Dict()
        self._rescale_fn: dict[str, RescaleFn] = {}

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
        self._init_rescaling(self._item_space)

    def _init_rescaling(self, space: gym.spaces.Dict) -> None:
        if self.normalize:
            for key, value in space.items():
                if not isinstance(value, gym.spaces.Box):
                    continue
                if np.issubdtype(value.dtype, np.floating):
                    if np.all(value.low == -1) and np.all(value.high == 1):
                        continue
                    self._rescale_fn[key] = normalize(value.low, value.high,
                                                      -1, 1)
                    #dtype = cast('np.typing.NDArray[np.floating[Any]]',
                    #value.dtype)
                    space[key] = gym.spaces.Box(low=-1,
                                                high=1,
                                                shape=value.shape,
                                                dtype=FloatType)
        else:
            self._rescale_fn.clear()

    def _make_state_space(self) -> gym.spaces.Dict:
        ds: dict[str, gym.Space[Any]] = {}
        if self.include_target_direction:
            ds['ego_target_direction'] = gym.spaces.Box(-1,
                                                        1, (2, ),
                                                        dtype=FloatType)
            if self.include_target_direction_validity:
                ds['ego_target_direction_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
        if self.include_target_distance:
            ds['ego_target_distance'] = gym.spaces.Box(
                0, self.max_target_distance, (1, ), dtype=FloatType)
            if self.include_target_distance_validity:
                ds['ego_target_distance_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
        if self.include_target_orientation:
            ds['ego_target_orientation'] = gym.spaces.Box(-1,
                                                          1, (2, ),
                                                          dtype=FloatType)
            if self.include_target_orientation_validity:
                ds['ego_target_orientation_valid'] = gym.spaces.Box(
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
        if self.include_position:
            high = []
            low = []
            if self.max_x > self.min_x:
                self.include_x = True
                high.append(self.max_x)
                low.append(self.min_x)
            else:
                self.include_x = False
            if self.max_y > self.min_y:
                self.include_y = True
                high.append(self.max_y)
                low.append(self.min_y)
            else:
                self.include_y = False
            if len(high):
                ds['ego_position'] = gym.spaces.Box(np.asarray(low),
                                                    np.asarray(high),
                                                    (len(high), ),
                                                    dtype=FloatType)
            else:
                self.include_position = False
        if self.include_orientation:
            ds['ego_orientation'] = gym.spaces.Box(-1,
                                                   1, (2, ),
                                                   dtype=FloatType)

        return gym.spaces.Dict(ds)

    def _rescale(self, obs: dict[str, Array]) -> None:
        for key, value in obs.items():
            if key in self._rescale_fn:
                obs[key] = self._rescale_fn[key](value)

    def get_observation(self, behavior: core.Behavior | None,
                        buffers: Mapping[str, core.Buffer]) -> Observation:
        rs = self._get_state_observations(behavior)
        if self.ignore_keys:
            rs = {k: v for k, v in rs.items() if k not in self.ignore_keys}
        if self.flat_values and not self.should_flatten_observations:
            rs.update((k, b.data.flatten()) for k, b in buffers.items()
                      if k not in self.ignore_keys)
        else:
            rs.update((k, b.data) for k, b in buffers.items()
                      if k not in self.ignore_keys)
        self._rescale(rs)
        if not self.should_flatten_observations:
            return rs
        vs = cast("Array", gym.spaces.flatten(self._item_space, rs))
        if self._dtype:
            vs = vs.astype(self._dtype)
        return vs

    @property
    def should_flatten_observations(self) -> bool:
        return self.flat or self.history > 1

    def _make_item_space(self,
                         sensing_space: gym.spaces.Dict) -> gym.spaces.Dict:
        if self.flat_values and not self.should_flatten_observations:
            sensing_space = flatten_dict_space(sensing_space)
        if self.ignore_keys or self.sort_keys or self.keys is not None:
            ks: Iterable[tuple[str, gym.spaces.Box]] = chain(
                cast('Iterable[tuple[str, gym.spaces.Box]]',
                     sensing_space.items()),
                cast('Iterable[tuple[str, gym.spaces.Box]]',
                     self._make_state_space().items()))
            if self.sort_keys:
                ks = sorted(ks)
            if self.keys is not None:
                rs = dict(ks)
                ks = {k: rs[k] for k in self.keys if k in rs}.items()
            return gym.spaces.Dict([(k, v) for k, v in ks
                                    if k not in self.ignore_keys])
        return gym.spaces.Dict(
            **sensing_space,  # type: ignore[arg-type]
            **self._make_state_space())  # type: ignore[arg-type]

    @property
    def space(self) -> gym.Space[Any]:
        if self.should_flatten_observations:
            flat_space: gym.spaces.Box = cast(
                "gym.spaces.Box", gym.spaces.flatten_space(self._item_space))
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
            if self.include_position:
                ps = behavior.position
                if not self.include_x:
                    ps = ps[1:]
                if not self.include_y:
                    ps = ps[:-1]
                rs['ego_position'] = ps
            if self.include_orientation:
                rs['ego_orientation'] = core.unit(behavior.orientation)
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
        if self.include_target_orientation:
            orientation = behavior.get_target_orientation(
                frame=core.Frame.relative)
            if orientation is not None:
                u = core.unit(orientation)
            else:
                u = core.zeros2()
            rs['ego_target_orientation'] = u
            if self.include_target_orientation_validity:
                value = 0 if orientation is None else 1
                rs['ego_target_orientation_valid'] = np.array([value],
                                                              np.uint8)
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
