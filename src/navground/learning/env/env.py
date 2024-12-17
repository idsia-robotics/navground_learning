from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias
import warnings

import gymnasium as gym
from gymnasium.envs.registration import register
from navground import sim

from ..config import (ActionConfig, ControlActionConfig,
                      DefaultObservationConfig, GroupConfig, ObservationConfig)
from ..indices import Indices
from ..internal.base_env import NavgroundBaseEnv
from ..policies.info_predictor import InfoPolicy
from ..rewards import NullReward
from ..types import Action, Bounds, Observation, Reward

BaseEnv: TypeAlias = gym.Env[Observation, Action]


class NavgroundEnv(NavgroundBaseEnv, BaseEnv):
    """
    This class describes an environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    Actions and observations relates to a single selected
    individual :py:class:`navground.sim.Agent`.

    The behavior is registered under the id ``"navground"``:

    >>> import gymnasium as gym
    >>> import navground.learning.env
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

    :param reward: The reward function to use. If none, it will default to constant zeros.

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

    :param stuck_timeout: The time to wait before considering an agent stuck.

    :param color: An optional color of the agent (only used for displaying)

    :param tag: An optional tag to be added to the agent (only used as metadata)
    """

    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 30}
    MAX_SEED = 2**31 - 1

    def __init__(self,
                 scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 agent_index: int = 0,
                 sensor: sim.Sensor | str | dict[str, Any] | None = None,
                 action: ActionConfig = ControlActionConfig(),
                 observation: ObservationConfig = DefaultObservationConfig(),
                 reward: Reward | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: Bounds | None = None,
                 terminate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0,
                 stuck_timeout: float = 1,
                 color: str = '',
                 tag: str = '') -> None:
        if reward is None:
            reward = NullReward()
        self.agent_index = agent_index
        group = GroupConfig(indices=Indices({agent_index}),
                            sensor=sensor,
                            reward=reward,
                            action=action,
                            observation=observation,
                            color=color,
                            tag=tag)
        NavgroundBaseEnv.__init__(
            self,
            groups=[group],
            scenario=scenario,
            time_step=time_step,
            max_duration=max_duration,
            bounds=bounds,
            terminate_outside_bounds=terminate_outside_bounds,
            render_mode=render_mode,
            render_kwargs=render_kwargs,
            realtime_factor=realtime_factor,
            stuck_timeout=stuck_timeout)

    def _init(self) -> None:
        super()._init()
        if self.scenario is not None:
            agent = self._possible_agents[self.agent_index]
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                configured = agent.is_configured(warn=True)
            if configured:
                self.observation_space = self._observation_space[
                    self.agent_index]
                self.action_space = self._action_space[self.agent_index]
            else:
                msgs = ', '.join([str(w.message) for w in ws])
                warnings.warn(
                    f"Configuration of agent at {self.agent_index} "
                    "is not complete. Check that the scenario spawns "
                    "the agent and that the action and observation "
                    f"configs have the required information: {msgs}",
                    stacklevel=0)
                self.observation_space = gym.spaces.Box(0, 1)
                self.action_space = gym.spaces.Box(0, 1)
            self.reward = agent.reward
        else:
            self.observation_space = gym.spaces.Box(0, 1)
            self.action_space = gym.spaces.Box(0, 1)
            self.action_config = None
            self.observation_config = None
            self.reward = None

    def _init_spec(
            self,
            scenario: sim.Scenario | str | dict[str, Any] | None) -> None:
        super()._init_spec(scenario)
        self._spec.pop('groups')
        group = next(iter(self.groups_config))
        self._spec['agent_index'] = self.agent_index
        self._spec['reward'] = group.reward
        self._spec['action'] = group.action
        self._spec['observation'] = group.observation
        self._spec['sensor'] = group.sensor

    @property
    def policy(self) -> InfoPolicy:
        """
        A policy that returns the action computed by the navground agent.

        :returns:   The policy.
        """
        return self.get_policy(self.agent_index)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """
        Conforms to :py:meth:`gymnasium.Env.reset`.

        It samples a new world from a scenario, runs one dry simulation step
        using :py:meth:`navground.sim.World.update_dry`.
        Then, it converts the agent's state to observations.
        and the command it would actuate to an action, which it includes
        in the ``info`` dictionary at key `"navground_action"``.
        """
        gym.Env.reset(self, seed=seed, options=options)
        NavgroundBaseEnv._reset(self, seed, options)
        obs = self.get_observations()
        infos = self.get_infos()
        return obs[self.agent_index], infos[self.agent_index]

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Action]]:
        """
        Conforms to :py:meth:`gymnasium.Env.step`.

        It converts the action to a command that the navground agent actuates.
        Then, it updates the world for one step, calling
        :py:meth:`navground.sim.World.update`.
        Finally, it converts the agent's state to observations,
        the command it would actuate to an action, which it includes
        in the ``info`` dictionary at key `"navground_action"`, and computes a reward.

        Termination is set when the agent completes the task, exits the boundary, or gets stuck.
        Truncation is set when the maximal duration has passed.
        """
        obs, reward, terminated, truncated, infos = NavgroundBaseEnv._step(
            self, {self.agent_index: action})
        return (obs[self.agent_index], reward[self.agent_index],
                terminated[self.agent_index], truncated[self.agent_index],
                infos[self.agent_index])


register(
    id="navground",
    entry_point="navground.learning.env:NavgroundEnv",
)
