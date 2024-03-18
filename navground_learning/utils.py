import numbers
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from navground import core, sim

Expert = Any


def make_simple_space(desc: core.BufferDescription) -> gym.Space:
    is_int = issubclass(desc.type.type, numbers.Integral)
    if desc.categorical and is_int and len(desc.shape) == 1:
        return gym.spaces.MultiDiscrete(nvec=[desc.high - desc.low + 1] *
                                        desc.shape[0],
                                        dtype=desc.type,
                                        start=[desc.low] * desc.shape[0])
    return gym.spaces.Box(desc.low, desc.high, desc.shape, dtype=desc.type)


def make_composed_space(
        value: Mapping[str, core.BufferDescription]) -> gym.Space:
    return gym.spaces.Dict(
        {k: make_simple_space(desc)
         for k, desc in value.items()})


def get_space_for_state(state: core.SensingState) -> gym.Space:
    return make_composed_space(
        {k: v.description
         for k, v in state.buffers.items()})


def get_space_for_sensor(sensor: sim.Sensor) -> gym.Space:
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
    scenario: sim.Scenario,
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

    :param max_speed: The upper bound of the speed.

    :param max_angular_speed: The upper bound of the angular speed.

    :param include_radius: Whether to include the own radius in the observations.

    :param max_radius: The upper bound of own radius.
                       Only relevant if ``include_radius=True``.

    :param fix_orientation: Whether to force the agent not to control orientation,
                            i.e., to not include the angular command in actions.

    :param dof: The degrees of freedom of the agent kinematics.
    """

    flat: bool = False
    history: int = 1
    include_target_distance: bool = True
    max_target_distance: float = np.inf
    include_target_direction: bool = True
    include_velocity: bool = False
    max_speed: float = np.inf
    max_angular_speed: float = np.inf
    include_radius: bool = False
    max_radius: float = np.inf
    fix_orientation: bool = False
    dof: int = 3
    _sensing_space: gym.spaces.Space = field(default_factory=gym.spaces.Dict,
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
        return self.fix_orientation and self.dof > 2

    @property
    def should_flatten_observations(self):
        return self.flat or self.history > 1

    @property
    def action_size(self) -> int:
        if self.dof == 2 or self.fix_orientation:
            return 2
        return 3

    def _use_behavior(self,
                      behavior: core.Behavior,
                      set_sensor: bool = False) -> None:
        self.max_speed = behavior.kinematics.max_speed
        self.max_angular_speed = behavior.kinematics.max_angular_speed
        self.max_target_distance = behavior.horizon
        self.dof = behavior.kinematics.dof

    def _use_scenario(self,
                      scenario: sim.Scenario,
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
        ds = {}
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
                                                (self.dof - 1, ),
                                                dtype=np.float64)
        if self.include_radius:
            ds['ego_radius'] = gym.spaces.Box(0,
                                              self.max_radius, (1, ),
                                              dtype=np.float64)
        return gym.spaces.Dict(ds)

    def _update_observation_space(self) -> None:
        self._observation_item_space = gym.spaces.Dict(
            **self._sensing_space, **self._compute_state_space())
        self._observation_space = self._compute_observation_space()

    def _compute_observation_space(self) -> gym.Space:
        space = self._observation_item_space
        if self.flat:
            space = gym.spaces.flatten_space(space)
        if self.history > 1:
            low = np.repeat(space.low[np.newaxis, ...], self.history, axis=0)
            high = np.repeat(space.high[np.newaxis, ...], self.history, axis=0)
            space = gym.spaces.Box(low=low,
                                   high=high,
                                   dtype=space.dtype or np.float64)
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
                  scenario: sim.Scenario | None = None,
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
            has_set_sensor = self._use_behavior(behavior,
                                                set_sensor=not has_set_sensor)
        if scenario:
            self._use_scenario(scenario, index, set_sensor=not has_set_sensor)
        self._update_observation_space()
        return self

    # DONE(Jerome): we were not rescaling the speed
    def get_cmd_from_action(self, action: np.ndarray) -> core.Twist2:
        if self.dof == 2 or not self.should_fix_orientation:
            angular_speed = action[-1] * self.max_angular_speed
        else:
            angular_speed = 0
        if self.dof == 2:
            velocity = np.array((action[0] * self.max_speed, 0))
        else:
            velocity = action[:2] * self.max_speed
        twist = core.Twist2(velocity=velocity,
                            angular_speed=angular_speed,
                            frame=core.Frame.relative)
        # DONE(Jerome): move away from here ... it requires a kinematics
        # cmd = self._behavior.feasible_twist(twist, frame)
        return twist

    def get_action(self, behavior: core.Behavior) -> np.ndarray:
        cmd = behavior.get_actuated_twist(core.Frame.relative)

        if self.max_speed:
            v = cmd.velocity[:2] / self.max_speed
        else:
            v = np.zeros(2)
        if self.max_angular_speed:
            w = cmd.angular_speed / self.max_angular_speed
        else:
            w = 0
        if self.dof == 2:
            return np.array([v[0], w], np.float64)
        elif self.should_fix_orientation:
            return v
        return np.array([*v, w], np.float64)

    def get_observations(
            self, behavior: core.Behavior,
            state: core.SensingState) -> Dict[str, np.ndarray] | np.ndarray:
        rs = {k: v.data for k, v in state.buffers.items()}
        if self.include_velocity:
            v = core.to_relative(behavior.velocity, behavior.pose)
            if self.dof == 2:
                rs['ego_velocity'] = v[:1]
            else:
                rs['ego_velocity'] = v
        if self.include_radius:
            rs['ego_radius'] = np.array([behavior.radius], dtype=np.float64)
        if self.include_target:
            p = get_relative_target_position(behavior)
            dist = np.linalg.norm(p)
            if dist > 0:
                p = p / dist
            if self.include_target_direction:
                rs['ego_target_direction'] = p
            if self.include_target_distance:
                rs['ego_target_distance'] = np.array(
                    [min(dist, behavior.horizon)])
        if not self.should_flatten_observations:
            return rs
        return gym.spaces.flatten(self._observation_item_space, rs)


def get_expert(config: GymAgentConfig) -> Expert:
    """
    The policy that extracts the navground action from the info dictionary
    :param      config:  The configuration

    :returns:   The policy.
    """

    # (obs, state, dones, infos) -> (actions, state)
    def expert(obs, state, done, info=None):
        acts = [
            i.get('navground_action', np.zeros(config.action_size))
            for i in info
        ], None
        return acts

    expert.observation_space = config.observation_space
    expert.action_space = config.action_space

    return expert


class GymAgent:

    def __init__(self, config: GymAgentConfig):
        self.config = config
        if config.history > 1:
            self._stack = deque(maxlen=config.history)
        else:
            self._stack = None

    def get_cmd_from_action(self, action: np.ndarray) -> core.Twist2:
        return self.config.get_cmd_from_action(action)

    def get_action(self, behavior: core.Behavior) -> np.ndarray:
        return self.config.get_action(behavior)

    def update_observations(
            self, behavior: core.Behavior,
            state: core.SensingState) -> Dict[str, np.ndarray] | np.ndarray:
        fs = self.config.get_observations(behavior, state)
        if self._stack is None:
            return fs
        self._stack.append(fs)
        while len(self._stack) < self.config.history:
            self._stack.append(fs)
        return np.asarray(self._stack)

    def reset(self) -> None:
        if self._stack:
            self._stack.clear()
