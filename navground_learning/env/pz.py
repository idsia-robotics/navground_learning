from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pettingzoo as pz
from navground import sim

from ..config import GroupConfig, Indices, WorldConfig
from ..core import ActionConfig, ControlActionConfig, ObservationConfig
from ..reward import NullReward, Reward
from .base import NavgroundBaseEnv


class MultiAgentNavgroundEnv(NavgroundBaseEnv, pz.ParallelEnv):

    metadata = {
        "name": "navground_ma",
        "render_modes": ["human", "rgb_array"],
        'render_fps': 30
    }
    MAX_SEED = 2**31 - 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        NavgroundBaseEnv.__init__(self, *args, **kwargs)
        self.possible_agents = list(self._possible_agents.keys())
        self.agents = list(self._agents.keys())

    def reset(self, seed=None, options=None):
        NavgroundBaseEnv._reset(self, seed, options)
        self.agents = list(self._agents.keys())
        return self.get_observations(), self.get_infos()

    def step(self, actions: dict[int, np.ndarray]):
        obs, rew, term, trunc, info = NavgroundBaseEnv._step(self, actions)
        self.agents = [
            index for index in self.agents
            if not (trunc.get(index, False) or term.get(index, False))
        ]
        return obs, rew, term, trunc, info

    def observation_space(self, agent: int) -> gym.Space:
        return self._observation_space[agent]

    def action_space(self, agent: int) -> gym.Space:
        return self._action_space[agent]


def parallel_env(scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 config: WorldConfig = WorldConfig(),
                 max_number_of_agents: int | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 terminate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0,
                 stuck_timeout: float = 1) -> MultiAgentNavgroundEnv:
    """
    Create a multi-agent PettingZoo environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param config: The configuration of the agents controlled by the environment.
                   All other agents are controlled solely by navground.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

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

    :returns:   The multi agent navground environment.
    """
    return MultiAgentNavgroundEnv(
        config=config,
        scenario=scenario,
        time_step=time_step,
        max_duration=max_duration,
        bounds=bounds,
        terminate_outside_bounds=terminate_outside_bounds,
        render_mode=render_mode,
        render_kwargs=render_kwargs,
        realtime_factor=realtime_factor,
        stuck_timeout=stuck_timeout)


def shared_parallel_env(
        scenario: sim.Scenario | str | dict[str, Any] | None = None,
        max_number_of_agents: int | None = None,
        agent_indices: Indices = None,
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
        realtime_factor: float = 1.0,
        stuck_timeout: float = 1) -> MultiAgentNavgroundEnv:
    """
    Create a multi-agent PettingZoo environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.
    All controlled agents share the same configuration.

    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

    :param agent_indices: The world indices of the agent to control.
                          All other agents are controlled solely by navground.

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

    :returns:   The multi agent navground environment.
    """
    config = GroupConfig(indices=agent_indices,
                         sensor=sensor,
                         reward=reward,
                         action=action,
                         observation=observation)
    return MultiAgentNavgroundEnv(
        config=WorldConfig([config]),
        scenario=scenario,
        time_step=time_step,
        max_duration=max_duration,
        bounds=bounds,
        terminate_outside_bounds=terminate_outside_bounds,
        render_mode=render_mode,
        render_kwargs=render_kwargs,
        realtime_factor=realtime_factor,
        stuck_timeout=stuck_timeout)
