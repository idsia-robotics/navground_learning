from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core, sim
from navground.core import FloatType

from ..types import Array
from .base import DataclassConfig, StateConfig

# TODO(Jerome): Complete with target state


@dc.dataclass(repr=False)
class DefaultStateConfig(DataclassConfig, StateConfig,
                         register_name="Default"):
    """
    This class configures which information from the behavior state to
    include in the global environement state.

    :param include_position: Whether observations include the position

    :param include_orientation: Whether observations include the orientation

    :param include_radius: Whether to include the radius in the observations.

    :param max_radius: Maximal radius.

    :param include_velocity: Whether to include the velocity

    :param include_angular_speed: Whether to include the angular_speed

    :param include_x: Whether observations include the x-coordinates

    :param include_y: Whether observations include the y-coordinates

    :param include_target_distance: Whether to include the target distance in the observations.

    :param max_target_distance: Maximal target distance

    :param include_target_distance_validity: Whether to include whether the target
                                             distance is valid in the observations.

    :param include_target_direction: Whether to include the target direction in the observations.

    :param include_target_direction_validity: Whether to include whether the target direction
                                              is valid in the observations.

    :param include_target_speed: Whether to include the target speed in the observations.

    :param include_target_angular_speed: Whether to include the target angular speed
                                         in the observations.
    """

    include_position: bool = False
    """Whether observations include the position."""
    include_orientation: bool = False
    """Whether observations include the orientation."""
    include_radius: bool = False
    """Whether observations include the radius."""
    max_radius: float = math.inf
    """Maximal radius."""
    include_velocity: bool = False
    """Whether to include the velocity."""
    include_angular_speed: bool = False
    """Whether to include the angular_speed."""
    include_x: bool = True
    """Whether observations include the x-coordinates"""
    include_y: bool = True
    """Whether observations include the y-coordinates"""
    use_absolute_frame: bool = True
    """Whether to use the absolute frame for velocity and target direction"""
    include_target_orientation: bool = False
    """TODO"""
    include_target_orientation_validity: bool = False
    """TODO"""
    include_target_distance: bool = False
    """Whether observations include the target direction."""
    max_target_distance: float = math.inf
    """Maximal target distance"""
    include_target_distance_validity: bool = False
    """Whether observations include the validity of the target direction."""
    include_target_direction: bool = False
    """Whether observations include the target direction."""
    include_target_direction_validity: bool = False
    """Whether observations include whether the validity of the target direction."""
    include_target_speed: bool = False
    """Whether observations include the target speed."""
    include_target_angular_speed: bool = False
    """Whether observations include the target angular speed."""
    include_all: dc.InitVar[bool] = False
    """Whether to include all fields"""

    def __post_init__(self, include_all: bool) -> None:
        super().__post_init__()
        if include_all:
            self.include_position = True
            self.include_orientation = True
            self.include_radius = True
            self.include_velocity = True
            self.include_angular_speed = True
            self.include_target_distance = True
            self.include_target_direction = True
            self.include_target_orientation = True
            self.include_target_speed = True
        self._dict_space: gym.spaces.Dict = gym.spaces.Dict()

    def get_space(self, world: sim.World) -> gym.spaces.Box:
        return cast('gym.spaces.Box',
                    gym.spaces.flatten_space(self._get_dict_space(world)))

    def get_state(self, world: sim.World) -> Array:
        rs = self._get_dict_state(world)
        vs = cast('Array', gym.spaces.flatten(self._get_dict_space(world), rs))
        if self._dtype:
            vs = vs.astype(self._dtype)
        return vs

    def _get_dict_space(self, world: sim.World) -> gym.spaces.Dict:
        if not self._dict_space:
            self._dict_space = self._make_dict_space(world)
        return self._dict_space

    def _make_dict_space(self, world: sim.World) -> gym.spaces.Dict:
        ds: dict[str, gym.Space[Any]] = {}
        num = len(world.agents)
        if self.include_velocity or self.include_target_speed:
            max_speeds = np.array([[agent.kinematics.max_speed]
                                   for agent in world.agents
                                   if agent.kinematics])
            if self.include_velocity:
                if self.use_absolute_frame:
                    max_vels = np.stack([max_speeds, max_speeds])
                    if not self.include_x:
                        max_vels = max_vels[1:]
                    if not self.include_y:
                        max_vels = max_vels[:1]
                else:
                    max_vels = np.array([[agent.kinematics.max_speed] *
                                         (agent.kinematics.dof() - 1)
                                         for agent in world.agents
                                         if agent.kinematics]).flatten()
                ds['velocity'] = gym.spaces.Box(-max_vels,
                                                max_vels,
                                                dtype=FloatType)
            if self.include_target_speed:
                ds['target_speed'] = gym.spaces.Box(-max_speeds, max_speeds)
        if self.include_angular_speed or self.include_target_angular_speed:
            max_angular_speeds = np.array([
                agent.kinematics.max_angular_speed for agent in world.agents
                if agent.kinematics
            ])
            space = gym.spaces.Box(-max_angular_speeds,
                                   max_angular_speeds,
                                   dtype=FloatType)
            if self.include_angular_speed:
                ds['angular_speed'] = space
            if self.include_target_angular_speed:
                ds['target_angular_speed'] = space
        if self.include_position:
            bl = world.bounding_box.p1
            tr = world.bounding_box.p2
            if not self.include_x:
                bl = bl[1:]
                tr = tr[1:]
            if not self.include_y:
                bl = bl[:1]
                tr = tr[:1]
            # TODO: use np.repeat
            bls = np.array([bl for _ in world.agents])
            trs = np.array([tr for _ in world.agents])
            ds['position'] = gym.spaces.Box(bls, trs, dtype=FloatType)
        if self.include_orientation:
            ds['orientation'] = gym.spaces.Box(-1,
                                               1, (num, 2),
                                               dtype=FloatType)
        if self.include_radius:
            if not math.isfinite(self.max_radius):
                self.max_radius = max(agent.radius for agent in world.agents)
            ds['radius'] = gym.spaces.Box(0,
                                          self.max_radius, (num, ),
                                          dtype=FloatType)
        if self.include_target_direction:
            dims = self._dims() if self.use_absolute_frame else 2
            if dims == 0:
                self.include_target_direction = False
            else:
                ds['target_direction'] = gym.spaces.Box(-1,
                                                        1, (num, dims),
                                                        dtype=FloatType)
                if self.include_target_direction_validity:
                    ds['target_direction_valid'] = gym.spaces.Box(
                        0, 1, (num, ), dtype=np.uint8)
        if self.include_target_distance:
            self.max_target_distance = min(
                self.max_target_distance,
                float(
                    np.linalg.norm(world.bounding_box.p1 -
                                   world.bounding_box.p2)))
            ds['target_distance'] = gym.spaces.Box(0,
                                                   self.max_target_distance,
                                                   (1, ),
                                                   dtype=FloatType)
            if self.include_target_distance_validity:
                ds['target_distance_valid'] = gym.spaces.Box(0,
                                                             1, (num, ),
                                                             dtype=np.uint8)
        if self.include_target_orientation:
            dims = 2
            if dims == 0:
                self.include_target_orientation = False
            else:
                ds['target_orientation'] = gym.spaces.Box(-1,
                                                          1, (num, dims),
                                                          dtype=FloatType)
                if self.include_target_orientation_validity:
                    ds['target_orientation_valid'] = gym.spaces.Box(
                        0, 1, (1, ), dtype=np.uint8)
        return gym.spaces.Dict(ds)

    def _get_dict_state(self, world: sim.World) -> dict[str, Array]:
        rs: dict[str, Array] = {}
        if self.include_position:
            rs['position'] = np.array([
                self._get_components(agent.position) for agent in world.agents
            ])
        if self.include_orientation:
            theta = np.array([agent.orientation for agent in world.agents])
            rs['orientation'] = np.concatenate(
                [np.cos(theta), np.sin(theta)], axis=0)
        if self.include_radius:
            rs['radius'] = np.array([agent.radius for agent in world.agents])
        if self.include_velocity:
            if self.use_absolute_frame:
                rs['velocity'] = np.array([
                    self._get_components(agent.velocity)
                    for agent in world.agents if agent.kinematics
                ])
            else:
                rs['velocity'] = np.concatenate([
                    agent.twist.relative(
                        agent.pose).velocity[:(agent.kinematics.dof() - 1)]
                    for agent in world.agents if agent.kinematics
                ])
        if self.include_angular_speed:
            rs['angular_speed'] = np.array(
                [agent.twist.angular_speed for agent in world.agents])
        if self.include_target_distance:
            values: list[float] = []
            validity: list[bool] = []
            for agent in world.agents:
                if agent.behavior:
                    distance = agent.behavior.get_target_distance()
                    values.append(min(distance or 0, self.max_target_distance))
                    validity.append(distance is not None)
                else:
                    values.append(0)
                    validity.append(False)
            rs['target_distance'] = np.array(values, dtype=FloatType)
            if self.include_target_distance_validity:
                rs['target_distance_valid'] = np.array(validity, np.uint8)
        if self.include_target_orientation:
            v_values: list[core.Vector2] = []
            validity = []
            for agent in world.agents:
                if agent.behavior:
                    orientation = agent.behavior.get_target_orientation(
                        frame=core.Frame.absolute if self.
                        use_absolute_frame else core.Frame.relative)
                    v_values.append(
                        core.unit(orientation) if orientation else core.zeros2())
                    validity.append(orientation is not None)
                else:
                    v_values.append(core.zeros2())
                    validity.append(False)
            rs['target_orientation'] = np.array(v_values, dtype=FloatType)
            rs['target_orientation_valid'] = np.array(validity, np.uint8)
        if self.include_target_direction:
            v_values = []
            validity = []
            for agent in world.agents:
                if agent.behavior:
                    direction = agent.behavior.get_target_direction(
                        frame=core.Frame.absolute if self.
                        use_absolute_frame else core.Frame.relative)
                    v = direction if direction is not None else core.zeros2()
                    if self.use_absolute_frame:
                        v = self._get_components(v)
                    v_values.append(v)
                else:
                    v_values.append(core.zeros2())
                    validity.append(False)
            rs['target_direction'] = np.array(v_values, dtype=FloatType)
            rs['target_direction_valid'] = np.array(validity, np.uint8)
        if self.include_target_speed:
            values = []
            for agent in world.agents:
                if agent.behavior:
                    values.append(agent.behavior.get_target_speed())
                else:
                    values.append(0)
            rs['target_speed'] = np.array(values, dtype=FloatType)
        if self.include_target_angular_speed:
            values = []
            for agent in world.agents:
                if agent.behavior:
                    values.append(agent.behavior.get_target_angular_speed())
                else:
                    values.append(0)
            rs['target_angular_speed'] = np.array(values, dtype=FloatType)
        return rs

    def _get_components(self, vs: core.Vector2) -> Array:
        if not self.include_x:
            vs = vs[1:]
        if not self.include_y:
            vs = vs[:1]
        return vs

    def _dims(self) -> int:
        return int(self.include_x) + int(self.include_y)
