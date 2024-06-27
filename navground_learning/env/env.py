from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from navground import sim

from ..config import GroupConfig, WorldConfig
from ..core import ActionConfig, ControlActionConfig, ObservationConfig
from ..reward import NullReward, Reward
from .base import NavgroundBaseEnv


class NavgroundEnv(NavgroundBaseEnv, gym.Env):
    """
    This class describes an environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    Actions and observations relates to a single selected
    individual :py:class:`navground.sim.Agent`.

    The behavior is registered under the id ``"navground"``:

    >>> import gymnasium as gym
    >>> import navground_learning.env
    >>> from navground import sim
    >>>
    >>> scenario = sim.load_scenario(...)
    >>> env = gym.make("navground", scenario=scenario)

    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param agent_index: The world index of the selected agent,
                        must be smaller than the number of agents.

    :param sensor: A sensor to produce observations for the selected agents.
                   If a :py:class:`str`, it will be interpreted as the YAML
                   representation of a sensor.
                   If a :py:class:`dict`, it will be dumped to YAML and
                   then treated as a :py:class:`str`.
                   If None, it will use the agents' own state estimation, if a sensor.

    :param action: The configuration of the action space to use.

    :param observation: The configuration of the observation space to use.

    :param reward: The reward function to use.

    :param time_step: The simulation time step applied at every :py:meth:`step`.

    :param max_duration: If positive, it will signal a truncation after this simulated time.

    :param terminate_outside_bounds: Whether to terminate when an agent exit the bounds

    :param bounds: The area to render and a fence for truncating processes when agents exit it.

    :param render_mode: The render mode.
                        If `"human"`, it renders a simulation in real time via
                        websockets (see :py:class:`navground.sim.ui.WebUI`).
                        If `"rgb_array"`, it uses
                        :py:func:`navground.sim.ui.render.image_for_world`
                        to render the world on demand.

    :param render_kwargs: Arguments passed to :py:func:`navground.sim.ui.render.image_for_world`

    :param realtime_factor: a realtime factor for `render_mode="human"`: larger values
                            speed up the simulation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 30}
    MAX_SEED = 2**31 - 1

    def __init__(self,
                 scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 agent_index: int = 0,
                 sensor: sim.Sensor | str | dict[str, Any] | None = None,
                 action: ActionConfig = ControlActionConfig(),
                 observation: ObservationConfig = ObservationConfig(),
                 reward: Reward = NullReward(),
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 terminate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0) -> None:

        config = GroupConfig(indices=[agent_index],
                             sensor=sensor,
                             reward=reward,
                             action=action,
                             observation=observation)
        NavgroundBaseEnv.__init__(
            self,
            config=WorldConfig([config]),
            scenario=scenario,
            time_step=time_step,
            max_duration=max_duration,
            bounds=bounds,
            terminate_outside_bounds=terminate_outside_bounds,
            render_mode=render_mode,
            render_kwargs=render_kwargs,
            realtime_factor=realtime_factor)
        self._spec.pop('config')
        self._spec['agent_index'] = agent_index
        self._spec['reward'] = reward
        self._spec['action'] = action
        self._spec['observation'] = observation
        self._spec['sensor'] = sensor
        self.agent_index = agent_index
        if scenario is not None:
            agent = self._possible_agents[agent_index]
            self.observation_space = self._observation_space[agent_index]
            self.action_space = self._action_space[agent_index]
            if agent.gym:
                self.action_config = agent.gym.action_config
                self.observation_config = agent.gym.observation_config
            self.reward = agent.reward
        else:
            self.observation_space = gym.spaces.Box(0, 1)
            self.action_space = gym.spaces.Box(0, 1)
            self.reward = None

    @property
    def asdict(self) -> dict[str, Any]:
        rs = NavgroundBaseEnv.asdict.fget(self)  # type: ignore
        config = rs.pop('config')['groups'][0]
        rs['agent_index'] = config.pop('indices')[0]
        rs.update(config)
        return rs

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        NavgroundBaseEnv._reset(self, seed, options)
        obs = self.get_observations()
        infos = self.get_infos()
        return obs[self.agent_index], infos[self.agent_index]

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, infos = NavgroundBaseEnv._step(
            self, {self.agent_index: action})
        return (obs[self.agent_index], reward[self.agent_index],
                terminated[self.agent_index], truncated[self.agent_index],
                infos[self.agent_index])


register(
    id="navground",
    entry_point="navground_learning.env:NavgroundEnv",
)
