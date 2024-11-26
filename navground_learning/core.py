import abc
import copy
import dataclasses as dc
import math
import numbers
import time
from collections import deque
from collections.abc import Mapping
from typing import Any, Optional, cast

import gymnasium as gym  # type: ignore
import numpy as np
from navground import core, sim

from .reward import Reward


def make_simple_space(desc: core.BufferDescription) -> gym.Space:
    is_int = issubclass(desc.type.type, numbers.Integral)
    if is_int and desc.low == 0 and desc.high == 1:
        return gym.spaces.MultiBinary(desc.shape)
    if desc.categorical and is_int and len(desc.shape) == 1:
        return gym.spaces.MultiDiscrete(
            nvec=[int(desc.high - desc.low + 1)] * desc.shape[0],
            dtype=desc.type,  # type: ignore
            start=[int(desc.low)] * desc.shape[0])
    return gym.spaces.Box(desc.low, desc.high, desc.shape,
                          dtype=desc.type)  # type: ignore


def make_composed_space(
        value: Mapping[str, core.BufferDescription]) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        k: make_simple_space(desc)
        for k, desc in value.items()
    })


def get_space_for_state(state: core.SensingState) -> gym.spaces.Dict:
    return make_composed_space({
        k: v.description
        for k, v in state.buffers.items()
    })


def get_space_for_sensor(sensor: sim.Sensor) -> gym.spaces.Dict:
    return make_composed_space(sensor.description)


def get_relative_target_position(behavior: core.Behavior) -> core.Vector2:
    if behavior.target.position is None:
        return np.zeros(2, dtype=np.float64)
    return core.to_relative(behavior.target.position - behavior.position,
                            behavior.pose)


def to_py(value: np.ndarray) -> Any:
    try:
        return value.item()
    except ValueError:
        pass
    return value.tolist()


class SyncClock:

    def __init__(self, time_step: float, factor: float = 1.0):
        self._last_time: Optional[float] = None
        self._period = time_step / factor

    def tick(self) -> None:
        now = time.monotonic()
        if self._last_time is not None:
            sleep_time = max(self._last_time + self._period - now, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.monotonic()
        self._last_time = now


@dc.dataclass
class Config:
    dtype: str = ''

    def __post_init__(self):
        if self.dtype:
            self._dtype = np.dtype(self.dtype)
            self.dtype = self._dtype.name
        else:
            self._dtype = None


@dc.dataclass
class ConfigWithKinematic(Config):
    max_speed: float = np.inf
    max_angular_speed: float = np.inf
    dof: int | None = None

    def configure(self, behavior: core.Behavior) -> None:
        kinematics = behavior.kinematics
        if not kinematics:
            return
        if not math.isfinite(self.max_speed):
            self.max_speed = kinematics.max_speed
        if not math.isfinite(self.max_angular_speed):
            self.max_angular_speed = kinematics.max_angular_speed
        if self.dof is None:
            self.dof = kinematics.dof()


@dc.dataclass
class ObservationConfig(ConfigWithKinematic):
    """

    Configuration of observations produced consumed
    by a :py:class:`navground.sim.Agent`
    in a :py:class:`navground_learning.env.NavgroundEnv`.

    :param flat: Whether to use a flat observation space

    :param history: The size of the queue to use for observations.
                    If larger than 1, recent observations will be stacked,
                    and then flattened.

    :param include_target_distance: Whether to include the target distance in the observations.

    :param include_target_distance_validity: Whether to include whether the target distnace is valid in the observations.

    :param max_target_distance: The upper bound of target distance.
                                Only relevant if ``include_target_distance=True``

    :param include_target_direction: Whether to include the target direction in the observations.

    :param include_target_direction_validity: Whether to include whether the target direction is valid in the observations.

    :param include_velocity: Whether to include the current velocity in the observations.

    :param include_angular_speed: Whether to include the current angular_speed in the observations.

    :param include_target_speed: Whether to include the target speed in the observations.

    :param include_target_angular_speed: Whether to include the target angular speed in the observations.

    :param max_speed: The upper bound of the speed.

    :param max_angular_speed: The upper bound of the angular speed.

    :param include_radius: Whether to include the own radius in the observations.

    :param max_radius: The upper bound of own radius.
                       Only relevant if ``include_radius=True``.

    :param use_wheels: Whether observation uses wheel speeds instead of body velocity.
                       Only effective if the agent uses a wheeled kinematics.

    """

    flat: bool = False
    history: int = 1
    include_target_distance: bool = True
    include_target_distance_validity: bool = False
    max_target_distance: float = np.inf
    include_target_direction: bool = True
    include_target_direction_validity: bool = False
    include_velocity: bool = False
    include_angular_speed: bool = False
    include_radius: bool = False
    include_target_speed: bool = False
    include_target_angular_speed: bool = False
    max_radius: float = np.inf

    @property
    def should_flatten_observations(self):
        return self.flat or self.history > 1

    @property
    def is_configured(self) -> bool:
        if not math.isfinite(self.max_speed):
            if self.include_velocity or self.include_target_speed:
                return False
        if self.dof is None and self.include_velocity:
            return False
        if not math.isfinite(self.max_angular_speed):
            if self.include_angular_speed or self.include_target_angular_speed:
                return False
        if not math.isfinite(
                self.max_target_distance) and self.include_target_distance:
            return False
        if not math.isfinite(self.max_radius) and self.include_radius:
            return False
        return True

    def configure(self, behavior: core.Behavior) -> None:
        super().configure(behavior)
        if not math.isfinite(self.max_target_distance):
            self.max_target_distance = behavior.horizon
        if not math.isfinite(self.max_radius):
            self.max_radius = behavior.radius

    @property
    def state_space(self) -> gym.spaces.Dict:
        ds: dict[str, gym.spaces.Box] = {}
        if self.include_target_direction:
            ds['ego_target_direction'] = gym.spaces.Box(-1,
                                                        1, (2, ),
                                                        dtype=np.float64)
            if self.include_target_direction_validity:
                ds['ego_target_direction_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
                # gym.spaces.Discrete(n=2)
        if self.include_target_distance:
            ds['ego_target_distance'] = gym.spaces.Box(
                0, self.max_target_distance, (1, ), dtype=np.float64)
            if self.include_target_direction_validity:
                ds['ego_target_distance_valid'] = gym.spaces.Box(
                    0, 1, (1, ), dtype=np.uint8)
        if self.include_velocity:
            if self.dof is None:
                raise ValueError("Set the DOF first")
            ds['ego_velocity'] = gym.spaces.Box(-self.max_speed,
                                                self.max_speed,
                                                (self.dof - 1, ),
                                                dtype=np.float64)
        if self.include_angular_speed:
            ds['ego_angular_speed'] = gym.spaces.Box(-self.max_angular_speed,
                                                     self.max_angular_speed,
                                                     (1, ),
                                                     dtype=np.float64)
        if self.include_radius:
            ds['ego_radius'] = gym.spaces.Box(0,
                                              self.max_radius, (1, ),
                                              dtype=np.float64)
        if self.include_target_speed:
            ds['ego_target_speed'] = gym.spaces.Box(0,
                                                    self.max_speed, (1, ),
                                                    dtype=np.float64)
        if self.include_target_angular_speed:
            ds['ego_target_angular_speed'] = gym.spaces.Box(
                0, self.max_angular_speed, (1, ), dtype=np.float64)
        return gym.spaces.Dict(ds)  # type: ignore

    def get_item_space(self, sensing_space: gym.spaces.Dict) -> gym.spaces.Dict:
        return gym.spaces.Dict(**sensing_space, **self.state_space)

    def get_space(self, item_space: gym.Space) -> gym.Space:
        if self.should_flatten_observations:
            flat_space: gym.spaces.Box = cast(
                gym.spaces.Box, gym.spaces.flatten_space(item_space))
            if self._dtype:
                flat_space.dtype = self._dtype
            if self.history > 1:
                low = np.repeat(flat_space.low[np.newaxis, ...],
                                self.history,
                                axis=0)
                high = np.repeat(flat_space.high[np.newaxis, ...],
                                 self.history,
                                 axis=0)
                flat_space = gym.spaces.Box(
                    low=low,
                    high=high,
                    dtype=item_space.dtype  # type: ignore
                    or np.float64)
            return flat_space
        return item_space

    def get_observations(
            self, behavior: core.Behavior | None,
            buffers: Mapping[str, core.Buffer],
            item_space: gym.Space) -> dict[str, np.ndarray] | np.ndarray:
        rs = {k: b.data for k, b in buffers.items()}
        if behavior:
            if self.include_velocity:
                v = core.to_relative(behavior.velocity, behavior.pose)
                if self.dof == 2:
                    rs['ego_velocity'] = v[:1]
                else:
                    rs['ego_velocity'] = v
            if self.include_angular_speed:
                rs['ego_angular_speed'] = np.array([behavior.angular_speed],
                                                   dtype=np.float64)
            if self.include_radius:
                rs['ego_radius'] = np.array([behavior.radius],
                                            dtype=np.float64)
            self._add_target(behavior, rs)
        if not self.should_flatten_observations:
            return rs
        vs = cast(np.ndarray, gym.spaces.flatten(item_space, rs))
        if self._dtype:
            vs = vs.astype(self._dtype)
        return vs

    def _add_target(self, behavior: core.Behavior,
                    rs: dict[str, np.ndarray]) -> None:
        if self.include_target_distance:
            distance = behavior.get_target_distance()
            rs['ego_target_distance'] = np.array(
                [min(distance or 0.0, self.max_target_distance)])
            if self.include_target_distance_validity:
                value = 0 if distance is None else 1
                rs['ego_target_distance_valid'] = np.array([value], np.uint8)
        if self.include_target_direction:
            e = behavior.get_target_direction(core.Frame.relative)
            rs['ego_target_direction'] = e if e is not None else np.zeros(2)
            if self.include_target_direction_validity:
                value = 0 if e is None else 1
                rs['ego_target_direction_valid'] = np.array([value], np.uint8)
        if self.include_target_speed:
            rs['ego_target_speed'] = np.array(
                [min(behavior.get_target_speed(), self.max_speed)])

        if self.include_target_angular_speed:
            rs['ego_target_angular_speed'] = np.array([
                min(behavior.get_target_angular_speed(),
                    self.max_angular_speed)
            ])

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

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs


@dc.dataclass
class ActionConfig(abc.ABC):

    @abc.abstractmethod
    def get_cmd_from_action(self, action: np.ndarray,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        ...

    @abc.abstractmethod
    def get_action(self, behavior: core.Behavior,
                   time_step: float) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def space(self) -> gym.Space:
        ...

    @abc.abstractmethod
    def configure(self, behavior: core.Behavior) -> None:
        ...

    @property
    @abc.abstractmethod
    def is_configured(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def asdict(self) -> dict[str, Any]:
        ...


class TemporaryBehaviorParams:

    def __init__(self, behavior: core.Behavior, params: dict[str, Any]):
        super().__init__()
        self._behavior = behavior
        self._params = params
        self._old_params: dict[str, Any] = {}

    def __enter__(self):
        for k, v in self._params.items():
            self._old_params[k] = getattr(self._behavior, k)
            setattr(self._behavior, k, v)

    def __exit__(self, exc_type, exc_value, exc_tb):
        for k, v in self._old_params.items():
            setattr(self._behavior, k, v)


def single_param_space(low: Any = 0,
                       high: Any = 1,
                       dtype: str = 'float',
                       discrete: bool = False,
                       normalized: bool = False) -> gym.spaces.Space:
    if discrete:
        return gym.spaces.Discrete(start=low, n=int(high - low))
    if normalized:
        low = -1
        high = 1
    return gym.spaces.Box(low, high, dtype=dtype)  # type: ignore


def param_space(params: dict[str, dict[str, Any]],
                normalized: bool = False) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        key:
        single_param_space(normalized=normalized, **value)
        for key, value in params.items()
    })


@dc.dataclass
class ModulationActionConfig(ActionConfig):

    params: dict[str, dict[str, Any]] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.param_space = param_space(self.params, normalized=True)

    @property
    def is_configured(self) -> bool:
        return True

    def normalize(self, key: str, value: Any) -> Any:
        param = self.params[key]
        if 'discrete' in param:
            return value
        low = param['low']
        high = param['high']
        return np.clip(-1, 1, -1 + 2 * (value - low) / (high - low))

    def de_normalize(self, key: str, value: Any) -> Any:
        param = self.params[key]
        if 'discrete' in param:
            return value
        low = param['low']
        high = param['high']
        return low + (value + 1) / 2 * (high - low)

    @property
    def space(self) -> gym.Space:
        return gym.spaces.flatten_space(self.param_space)

    def get_params_from_action(self, action: np.ndarray) -> dict[str, Any]:
        return {
            k: to_py(self.de_normalize(k, v))
            for k, v in gym.spaces.unflatten(self.param_space, action).items()
        }

    def get_cmd_from_action(self, action: np.ndarray,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        assert behavior is not None
        params = self.get_params_from_action(action)
        with TemporaryBehaviorParams(behavior, params):
            cmd = behavior.compute_cmd(time_step)
        return cmd

    def get_action(self, behavior: core.Behavior,
                   time_step: float) -> np.ndarray:
        params = {
            k: self.normalize(k, getattr(behavior, k))
            for k in self.param_space
        }
        return cast(np.ndarray, gym.spaces.flatten(self.param_space, params))

    def configure(self, behavior: core.Behavior) -> None:
        pass

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs


@dc.dataclass
class ControlActionConfig(ConfigWithKinematic, ActionConfig):
    """

    Configuration of actions consumed by a :py:class:`navground.sim.Agent`
    in a :py:class:`navground_learning.env.NavgroundEnv`.

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
    max_angular_acceleration: float = np.inf
    use_acceleration_action: bool = False
    fix_orientation: bool = False
    use_wheels: bool = False
    has_wheels: bool | None = None

    @property
    def should_use_wheels(self) -> bool:
        if self.has_wheels is None:
            raise ValueError("Set has_wheels first")
        return self.use_wheels and self.has_wheels

    @property
    def is_configured(self) -> bool:
        if self.use_wheels and self.has_wheels is None:
            return False
        if self.fix_orientation and self.dof is None:
            return False
        if self.use_acceleration_action:
            if not math.isfinite(self.max_acceleration) or self.dof is None:
                return False
            if not self.should_fix_orientation and not self.should_use_wheels and not math.isfinite(
                    self.max_angular_acceleration):
                return False
        else:
            if not math.isfinite(self.max_speed):
                return False
            if not self.should_fix_orientation and not self.should_use_wheels and not math.isfinite(
                    self.max_angular_speed):
                return False
        return True

    def configure(self, behavior: core.Behavior) -> None:
        super().configure(behavior)
        if self.has_wheels is None:
            self.has_wheels = behavior.kinematics.is_wheeled()

    @property
    def should_fix_orientation(self) -> bool:
        if self.dof is None:
            raise ValueError("Set the DOF first")
        return self.fix_orientation and self.dof > 2

    @property
    def should_flatten_observations(self):
        return self.flat or self.history > 1

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
    def space(self) -> gym.Space:
        """
        The configured action space

        :returns:  The action space
        """
        return gym.spaces.Box(-1,
                              1,
                              shape=(self.action_size, ),
                              dtype=self._dtype or np.float64)

    def _compute_wheels_value_from_action(self, action: np.ndarray,
                                          max_value: float) -> np.ndarray:
        return action * max_value

    def _compute_value_from_action(self, action: np.ndarray, max_value: float,
                                   max_angular_value: float) -> core.Twist2:
        if self.dof == 2 or not self.should_fix_orientation:
            angular_value = action[-1] * max_angular_value
        else:
            angular_value = 0
        if self.dof == 2:
            value = np.array((action[0] * max_value, 0))
        else:
            value = action[:2] * max_value
        twist = core.Twist2(velocity=value,
                            angular_speed=angular_value,
                            frame=core.Frame.relative)
        return twist

    # DONE(Jerome): we were not rescaling the speed
    def get_cmd_from_action(self, action: np.ndarray,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        if self.should_use_wheels:
            assert behavior is not None
            if self.use_acceleration_action:
                accs = self._compute_wheels_value_from_action(
                    action, self.max_acceleration)
                speeds = np.array(behavior.wheel_speeds) + accs * time_step
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
                           max_angular_value: float) -> np.ndarray:
        if max_value:
            v = value.velocity[:2] / max_value
        else:
            v = np.zeros(2)
        if max_angular_value:
            w = value.angular_speed / max_angular_value
        else:
            w = 0
        if self.dof == 2:
            return np.array([v[0], w], np.float64)
        elif self.should_fix_orientation:
            return v
        return np.array([*v, w], np.float64)

    def _action_from_wheels(self, values: np.ndarray,
                            max_value: float) -> np.ndarray:
        if max_value:
            return values / max_value
        return np.zeros(self.action_size, np.float64)

    def _compute_action_from_wheel_speeds(
            self, behavior: core.Behavior) -> np.ndarray:
        ws = np.array(behavior.actuated_wheel_speeds, np.float64)
        return self._action_from_wheels(ws, self.max_speed)

    def _compute_action_from_wheel_accelerations(
            self, behavior: core.Behavior, time_step: float) -> np.ndarray:
        ws = np.array(behavior.actuated_wheel_speeds, np.float64)
        cs = np.array(behavior.wheel_speeds)
        return self._action_from_wheels((ws - cs) / time_step,
                                        self.max_acceleration)

    def _compute_action_from_acceleration(self, behavior: core.Behavior,
                                          time_step: float) -> np.ndarray:
        cmd = behavior.get_actuated_twist(core.Frame.relative)
        twist = behavior.get_twist(core.Frame.relative)
        acc = core.Twist2(
            (cmd.velocity - twist.velocity) / time_step,
            (cmd.angular_speed - twist.angular_speed) / time_step)
        return self._action_from_twist(acc, self.max_acceleration,
                                       self.max_angular_acceleration)

    def _compute_action_from_velocity(self,
                                      behavior: core.Behavior) -> np.ndarray:
        cmd = behavior.get_actuated_twist(core.Frame.relative)
        return self._action_from_twist(cmd, self.max_speed,
                                       self.max_angular_speed)

    def get_action(self, behavior: core.Behavior,
                   time_step: float) -> np.ndarray:
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

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs


def get_state(behavior: core.Behavior | None,
              state: core.SensingState | None) -> core.SensingState | None:
    if state is None and behavior and isinstance(behavior.environment_state,
                                                 core.SensingState):
        return behavior.environment_state
    return state


class GymAgent:

    sensing_space: gym.spaces.Dict
    observation_item_space: gym.spaces.Dict
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    def __init__(self,
                 action: ActionConfig,
                 observation: ObservationConfig = ObservationConfig(),
                 behavior: core.Behavior | None = None,
                 state: core.SensingState | None = None):
        state = get_state(behavior, state)
        if state:
            self.sensing_space = get_space_for_state(state)
        else:
            self.sensing_space = gym.spaces.Dict({})
        self.observation_config = copy.copy(observation)
        self.action_config = copy.copy(action)
        if behavior:
            self.action_config.configure(behavior)
            self.observation_config.configure(behavior)
        self.action_space = self.action_config.space
        self.observation_item_space = self.observation_config.get_item_space(
            self.sensing_space)
        self.observation_space = self.observation_config.get_space(
            self.observation_item_space)
        self.init(behavior, state)

    def init(self, behavior: core.Behavior | None,
             state: core.SensingState | None) -> None:
        self._behavior = behavior
        state = get_state(behavior, state)
        if state:
            self._buffers = dict(state.buffers)
        else:
            self._buffers = {}
        history = self.observation_config.history
        if history > 1:
            self._stack: deque[dict[str, np.ndarray]
                               | np.ndarray] | None = deque(maxlen=history)
        else:
            self._stack = None

    def get_cmd_from_action(self, action: np.ndarray,
                            time_step: float) -> core.Twist2:
        return self.action_config.get_cmd_from_action(action, self._behavior,
                                                      time_step)

    def get_action(self, time_step: float) -> np.ndarray:
        assert self._behavior is not None
        return self.action_config.get_action(self._behavior, time_step)

    def update_observations(self) -> dict[str, np.ndarray] | np.ndarray:
        fs = self.observation_config.get_observations(
            self._behavior,
            self._buffers,
            item_space=self.observation_item_space)
        if self._stack is None:
            return fs
        self._stack.append(fs)
        while len(self._stack) < self.observation_config.history:
            self._stack.append(fs)
        return np.asarray(self._stack)

    # def reset(self) -> None:
    #     if self._stack:
    #         self._stack.clear()

    def __getstate__(self) -> tuple:
        return (self.action_space, self.sensing_space,
                self.observation_item_space, self.observation_space,
                self.action_config, self.observation_config)

    def __setstate__(self, state: tuple):

        (self.action_space, self.sensing_space, self.observation_item_space,
         self.observation_space, self.action_config,
         self.observation_config) = state
        self.init(None, None)


class Expert:

    key: str = 'navground_action'
    """
    A callable policy that extracts the navground action from the info dictionary
    :param      config:  The configuration
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space
        self._default = self.action_space.sample() * 0

    # (obs, state, dones, infos) -> (actions, state)
    def __call__(self,
                 obs,
                 state,
                 done,
                 info=None) -> tuple[list[np.ndarray], None]:
        acts = [i.get(Expert.key, self._default) for i in info], None
        return acts


@dc.dataclass
class Agent:
    gym: GymAgent | None = None
    reward: Reward | None = None
    navground: sim.Agent | None = None
    state: core.SensingState | None = None
    sensor: sim.Sensor | None = None
    policy: Any = None

    def get_sensor(self) -> sim.Sensor | None:
        if self.sensor:
            return self.sensor
        if self.navground and isinstance(self.navground.state_estimation,
                                         sim.Sensor):
            return self.navground.state_estimation
        return None

    def update_state(self, world: sim.World) -> None:
        if self.sensor and self.state and self.navground:
            self.sensor.update(self.navground, world, self.state)
