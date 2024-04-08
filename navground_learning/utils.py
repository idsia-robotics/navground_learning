import numbers
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, Deque, cast

import gymnasium as gym
import numpy as np
from navground import core, sim


def make_simple_space(desc: core.BufferDescription) -> gym.Space:
    is_int = issubclass(desc.type.type, numbers.Integral)
    if is_int and desc.low == 0 and desc.high == 1:
        return gym.spaces.MultiBinary(desc.shape)
    if desc.categorical and is_int and len(desc.shape) == 1:
        return gym.spaces.MultiDiscrete(nvec=[int(desc.high - desc.low + 1)] *
                                        desc.shape[0],
                                        dtype=desc.type,
                                        start=[int(desc.low)] * desc.shape[0])
    return gym.spaces.Box(desc.low, desc.high, desc.shape, dtype=desc.type)


def make_composed_space(
        value: Mapping[str, core.BufferDescription]) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {k: make_simple_space(desc)
         for k, desc in value.items()})


def get_space_for_state(state: core.SensingState) -> gym.spaces.Dict:
    return make_composed_space(
        {k: v.description
         for k, v in state.buffers.items()})


def get_space_for_sensor(sensor: sim.Sensor) -> gym.spaces.Dict:
    return make_composed_space(sensor.description)


def get_relative_target_position(behavior: core.Behavior) -> core.Vector2:
    if behavior.target.position is None:
        return np.zeros(2, dtype=np.float64)
    return core.to_relative(behavior.target.position - behavior.position,
                            behavior.pose)


# (obs, state, dones, infos) -> (actions, state)
# def expert(obs, state, done, info=None):
#     return [i.get('action', np.zeros(2)) for i in info], None


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


def get_behavior_and_sensor(
    scenario: sim._Scenario,
    index: int,
    sensor: sim.Sensor | None = None
) -> Tuple[core.Behavior | None, sim.Sensor | None]:
    world = sim.World()
    scenario.init_world(world)
    agent = world.agents[index]
    behavior = agent.behavior
    if sensor is None and isinstance(agent.state_estimation, sim.Sensor):
        sensor = agent.state_estimation
    return behavior, sensor


@dataclass
class GymAgentConfig:
    """

    Configuration of observations produced and actions consumed
    by a :py:class:`navground.sim.Agent`
    in a :py:class:`navground_learning.env.NavgroundEnv`.

    :param flat: Whether to use a flat observation space

    :param history: The size of the queue to use for observations.
                    If larger than 1, recent observations will be stacked,
                    and then flattened.

    :param include_target_distance: Whether to include the target distance in the observations.

    :param max_target_distance: The upper bound of target distance.
                                Only relevant if ``include_target_distance=True``

    :param include_target_direction: Whether to include the target direction in the observations.

    :param include_velocity: Whether to include the current velocity in the observations.

    :param include_angular_speed: Whether to include the current angular_speed in the observations.

    :param max_speed: The upper bound of the speed.

    :param max_angular_speed: The upper bound of the angular speed.

    :param include_radius: Whether to include the own radius in the observations.

    :param max_radius: The upper bound of own radius.
                       Only relevant if ``include_radius=True``.

    :param max_acceleration: The upper bound of the acceleration.

    :param max_angular_acceleration: The upper bound of the angular acceleration.

    :param use_acceleration_action: Whether actions are accelerations.
                                    If not set, actions are velocities.

    :param use_wheels: Whether action and observation uses wheel speeds/acceleration.
                        If not set, action and observation uses body speeds/acceleration.
                        Only effective if the agent uses a wheeled kinematics.

    :param fix_orientation: Whether to force the agent not to control orientation,
                            i.e., to not include the angular command in actions.

    """

    flat: bool = False
    history: int = 1
    include_target_distance: bool = True
    max_target_distance: float = np.inf
    include_target_direction: bool = True
    include_velocity: bool = False
    include_angular_speed: bool = False
    max_speed: float = np.inf
    max_angular_speed: float = np.inf
    include_radius: bool = False
    max_radius: float = np.inf
    max_acceleration: float = np.inf
    max_angular_acceleration: float = np.inf
    use_acceleration_action: bool = False
    use_wheels: bool = False
    fix_orientation: bool = False
    # The degrees of freedom of the agent kinematics.
    _dof: int = field(default=3, init=False, repr=False)
    # Whether the agent kinematic has wheels
    _has_wheels: bool = field(default=False, init=False, repr=False)
    _sensing_space: gym.spaces.Dict = field(default_factory=gym.spaces.Dict,
                                            init=False,
                                            repr=False)
    _observation_item_space: gym.spaces.Space = field(init=False, repr=False)
    _observation_space: gym.spaces.Space = field(init=False, repr=False)

    def __post_init__(self):
        self._update_observation_space()

    @property
    def include_target(self):
        return self.include_target_direction or self.include_target_distance

    @property
    def should_fix_orientation(self):
        return self.fix_orientation and self._dof > 2

    @property
    def should_flatten_observations(self):
        return self.flat or self.history > 1

    @property
    def action_size(self) -> int:
        if self.should_use_wheels:
            if self._dof == 2:
                return 2
            else:
                return 4
        if self._dof == 2 or self.fix_orientation:
            return 2
        return 3

    @property
    def should_use_wheels(self):
        return self.use_wheels and self._has_wheels

    def _use_behavior(self,
                      behavior: core.Behavior,
                      set_sensor: bool = False) -> None:
        self.max_speed = behavior.kinematics.max_speed
        self.max_angular_speed = behavior.kinematics.max_angular_speed
        self.max_target_distance = behavior.horizon
        self._dof = behavior.kinematics.dof
        self._has_wheels = behavior.kinematics.is_wheeled

    def _use_scenario(self,
                      scenario: sim._Scenario,
                      index: int,
                      set_sensor: bool = False) -> None:
        behavior, sensor = get_behavior_and_sensor(scenario, index)
        if behavior:
            self._use_behavior(behavior)
        if set_sensor and sensor:
            self._use_sensor(sensor)

    def _use_sensor(self, sensor: sim.Sensor) -> None:
        self._sensing_space = get_space_for_sensor(sensor)

    def _use_state(self, state: core.SensingState) -> None:
        self._sensing_space = get_space_for_state(state)

    def _compute_state_space(self) -> gym.spaces.Dict:
        ds: Dict[str, gym.spaces.Box] = {}
        if self.include_target_direction:
            ds['ego_target_direction'] = gym.spaces.Box(-1,
                                                        1, (2, ),
                                                        dtype=np.float64)
        if self.include_target_distance:
            ds['ego_target_distance'] = gym.spaces.Box(
                0, self.max_target_distance, (1, ), dtype=np.float64)
        if self.include_velocity:
            ds['ego_velocity'] = gym.spaces.Box(-self.max_speed,
                                                self.max_speed,
                                                (self._dof - 1, ),
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
        return gym.spaces.Dict(ds)  # type: ignore

    def _update_observation_space(self) -> None:
        self._observation_item_space = gym.spaces.Dict(
            **self._sensing_space, **self._compute_state_space())
        self._observation_space = self._compute_observation_space()

    def _compute_observation_space(self) -> gym.Space:
        space = self._observation_item_space
        if self.should_flatten_observations:
            flat_space: gym.spaces.Box = cast(gym.spaces.Box,
                                              gym.spaces.flatten_space(space))
            if self.history > 1:
                low = np.repeat(flat_space.low[np.newaxis, ...],
                                self.history,
                                axis=0)
                high = np.repeat(flat_space.high[np.newaxis, ...],
                                 self.history,
                                 axis=0)
                flat_space = gym.spaces.Box(low=low,
                                            high=high,
                                            dtype=space.dtype or np.float64)
            return flat_space
        return space

    @property
    def observation_space(self) -> gym.Space:
        """
        The configured observation space

        :returns:  The observation space
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """
        The configured action space

        :returns:  The action space
        """
        return gym.spaces.Box(-1,
                              1,
                              shape=(self.action_size, ),
                              dtype=np.float64)

    def configure(self,
                  scenario: sim._Scenario | None = None,
                  behavior: core.Behavior | None = None,
                  state: core.SensingState | None = None,
                  sensor: sim.Sensor | None = None,
                  index: int = 0) -> 'GymAgentConfig':
        """
        Configure for use by a specific navigation behavior and/or sensor.

        If a sensor or a state are provided, it updates the sensing space.
        If a behavior is provided, it inherits parameters like :py:attr:`max_speed`.
        If a scenario, it first sample a behavior (and possibly a sensor) associated
        with the agent at the given index, and then perform similar updates.

        :param      scenario:  The scenario
        :param      behavior:  The behavior
        :param      state:     The state
        :param      sensor:    The sensor
        :param      index:     The agent index in the scenario

        :returns:   The configuration specific to the provided navground objects.
        """
        has_set_sensor = False
        if sensor:
            self._use_sensor(sensor)
            has_set_sensor = True
        if state:
            self._use_state(state)
            has_set_sensor = True
        if behavior:
            self._use_behavior(behavior, set_sensor=not has_set_sensor)
        if scenario:
            self._use_scenario(scenario, index, set_sensor=not has_set_sensor)
        self._update_observation_space()
        return self

    def _compute_wheels_value_from_action(self, action: np.ndarray,
                                          max_value: float) -> np.ndarray:
        return action * max_value

    def _compute_value_from_action(self, action: np.ndarray, max_value: float,
                                   max_angular_value: float) -> core.Twist2:
        if self._dof == 2 or not self.should_fix_orientation:
            angular_value = action[-1] * max_angular_value
        else:
            angular_value = 0
        if self._dof == 2:
            value = np.array((action[0] * max_value, 0))
        else:
            value = action[:2] * max_value
        twist = core.Twist2(velocity=value,
                            angular_speed=angular_value,
                            frame=core.Frame.relative)
        return twist

    # DONE(Jerome): we were not rescaling the speed
    def get_cmd_from_action(self, action: np.ndarray, behavior: core.Behavior,
                            time_step: float) -> core.Twist2:
        if self.should_use_wheels:
            if self.use_acceleration_action:
                accs = self._compute_wheels_value_from_action(
                    action, self.max_acceleration)
                speeds = np.array(behavior.wheel_speeds) + accs * time_step
                return behavior.twist_from_wheel_speeds(speeds)
            speeds = self._compute_wheels_value_from_action(
                action, self.max_speed)
            return behavior.twist_from_wheel_speeds(speeds)
        if self.use_acceleration_action:
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
        if self._dof == 2:
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

    def get_observations(
        self, behavior: core.Behavior,
        buffers: Mapping[str,
                         core.Buffer]) -> Dict[str, np.ndarray] | np.ndarray:
        rs: Dict[str, np.ndarray] = {k: b.data for k, b in buffers.items()}
        if self.include_velocity:
            v = core.to_relative(behavior.velocity, behavior.pose)
            if self._dof == 2:
                rs['ego_velocity'] = v[:1]
            else:
                rs['ego_velocity'] = v
        if self.include_angular_speed:
            rs['ego_angular_speed'] = np.array([behavior.angular_speed],
                                               dtype=np.float64)
        if self.include_radius:
            rs['ego_radius'] = np.array([behavior.radius], dtype=np.float64)
        if self.include_target:
            self.update_target(behavior, rs)
        if not self.should_flatten_observations:
            return rs
        return cast(np.ndarray,
                    gym.spaces.flatten(self._observation_item_space, rs))

    def update_target(self, behavior: core.Behavior,
                      rs: Dict[str, np.ndarray]) -> None:
        if behavior.target.position is not None:
            p = core.to_relative(behavior.target.position - behavior.position,
                                 behavior.pose)
            dist = np.linalg.norm(p)
            if self.include_target_direction:
                if dist > 0:
                    p = p / dist
                rs['ego_target_direction'] = p
            if self.include_target_distance:
                rs['ego_target_distance'] = np.array(
                    [min(dist, behavior.horizon)])
        elif behavior.target.direction is not None and self.include_target_direction:
            rs['ego_target_direction'] = core.to_relative(
                behavior.target.direction, behavior.pose)


class Expert:
    """
    A callable policy that extracts the navground action from the info dictionary
    :param      config:  The configuration
    """

    def __init__(self, config: GymAgentConfig):
        self._default = np.zeros(config.action_size)
        self.observation_space = config.observation_space
        self.action_space = config.action_space

    # (obs, state, dones, infos) -> (actions, state)
    def __call__(self,
                 obs,
                 state,
                 done,
                 info=None) -> Tuple[list[np.ndarray], None]:
        acts = [i.get('navground_action', self._default) for i in info], None
        return acts


class GymAgent:

    def __init__(self, config: GymAgentConfig, behavior: core.Behavior,
                 state: core.SensingState | None):
        self.config = config.configure(behavior=behavior, state=state)
        if config.history > 1:
            self._stack: Deque[Dict[str, np.ndarray]
                               | np.ndarray] | None = deque(
                                   maxlen=config.history)
        else:
            self._stack = None
        if state:
            self._buffers = dict(state.buffers)
        else:
            self._buffers = {}
        self._behavior = behavior

    def get_cmd_from_action(self, action: np.ndarray,
                            time_step: float) -> core.Twist2:
        return self.config.get_cmd_from_action(action, self._behavior,
                                               time_step)

    def get_action(self, time_step: float) -> np.ndarray:
        return self.config.get_action(self._behavior, time_step)

    def update_observations(self) -> Dict[str, np.ndarray] | np.ndarray:
        fs = self.config.get_observations(self._behavior, self._buffers)
        if self._stack is None:
            return fs
        self._stack.append(fs)
        while len(self._stack) < self.config.history:
            self._stack.append(fs)
        return np.asarray(self._stack)

    def reset(self) -> None:
        if self._stack:
            self._stack.clear()
