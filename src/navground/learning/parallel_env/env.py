from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from navground import sim
from pettingzoo.utils.env import ParallelEnv

from ..config import (ActionConfig, ControlActionConfig,
                      DefaultObservationConfig, GroupConfig, ObservationConfig)
from ..env import BaseEnv, NavgroundEnv
from ..indices import Indices, IndicesLike
from ..internal.base_env import NavgroundBaseEnv, ResetReturn, StepReturn
from ..rewards import NullReward
from ..types import Action, Bounds, Observation, Reward

if TYPE_CHECKING:
    import gymnasium as gym

BaseParallelEnv: TypeAlias = ParallelEnv[int, Observation, Action]


class MultiAgentNavgroundEnv(NavgroundBaseEnv, BaseParallelEnv):
    """
    This class describes an environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    Actions and observations relates to one or more selected
    :py:class:`navground.sim.Agent`.

    We provide convenience functions to initialize the class
    :py:func:`parallel_env` and :py:func:`shared_parallel_env`
    to create the environment.
    """

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

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:
        """
        Conforms to :py:meth:`pettingzoo.utils.env.ParallelEnv.reset`.

        It samples a new world from a scenario, runs one dry simulation step
        using :py:meth:`navground.sim.World.update_dry`.
        Then, it converts the agents states to observations.
        and the commands they would actuate to actions, which it includes
        in the ``info`` dictionary at key `"navground_action"``.
        """
        NavgroundBaseEnv._reset(self, seed, options)
        self.agents = list(self._agents.keys())
        return self.get_observations(), self.get_infos()

    def step(self, actions: dict[int, Action]) -> StepReturn:
        """
        Conforms to :py:meth:`pettingzoo.utils.env.ParallelEnv.step`.

        It converts the actions to commands that the navground agents actuate.
        Then, it updates the world for one step,
        calling :py:meth:`navground.sim.World.update`.
        Finally, it converts the agents states to observations,
        the commands they would actuate to actions, which it includes
        in the ``info`` dictionary at key `"navground_action"`, and computes rewards.

        Termination for individual agents is set when they complete the task,
        exit the boundary, or get stuck.
        Truncation for all agents is set when the maximal duration has passed.
        """
        obs, rew, term, trunc, info = NavgroundBaseEnv._step(self, actions)
        self.agents = [
            index for index in self.agents
            if not (trunc.get(index, False) or term.get(index, False))
        ]
        return obs, rew, term, trunc, info

    def observation_space(self, agent: int) -> gym.Space[Any]:
        return self._observation_space[agent]

    def action_space(self, agent: int) -> gym.Space[Any]:
        return self._action_space[agent]


def parallel_env(scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 groups: Collection[GroupConfig] = tuple(),
                 max_number_of_agents: int | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: Bounds | None = None,
                 terminate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0,
                 stuck_timeout: float = 1) -> MultiAgentNavgroundEnv:
    """
    Create a multi-agent PettingZoo environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.


    >>> from navground.learning.parallel_env import parallel_env
    >>> from navground.learning import GroupConfig
    >>> from navground import sim
    >>>
    >>> scenario = sim.load_scenario(...)
    >>> groups = [GroupConfig(...), ...]
    >>> env = parallel_env(scenario=scenario, groups=groups)


    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param groups: The configuration of the agents controlled by the environment.
                   All other agents are controlled solely by navground.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

    :param time_step: The simulation time step applied
                      at every :py:meth:`MultiAgentNavgroundEnv.step`.

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

    :returns:   The multi agent navground environment.
    """
    return MultiAgentNavgroundEnv(
        groups=groups,
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
        indices: IndicesLike = Indices.all(),
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
        tag: str = '') -> MultiAgentNavgroundEnv:
    """
    Create a multi-agent PettingZoo environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.
    All controlled agents share the same configuration.


    >>> from navground.learning.parallel_env import shared_parallel_env
    >>> from navground import sim
    >>>
    >>> scenario = sim.load_scenario(...)
    >>> env = shared_parallel_env(scenario=scenario, ...)

    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

    :param indices: The world indices of the agent to control.
                    All other agents are controlled solely by navground.

    :param sensor: A sensor to produce observations for the selected agents.
                   If a :py:class:`str`, it will be interpreted as the YAML
                   representation of a sensor.
                   If a :py:class:`dict`, it will be dumped to YAML and
                   then treated as a :py:class:`str`.
                   If None, it will use the agents' own state estimation, if a sensor.

    :param action: The configuration of the action space to use.

    :param observation: The configuration of the observation space to use.

    :param reward: The reward function to use. If none, it will default to constant zeros.

    :param time_step: The simulation time step applied
                      at every :py:meth:`MultiAgentNavgroundEnv.step`.

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

    :param color: An optional color of the agents (only used for displaying)

    :param tag: An optional tag to be added to the agents (only used as metadata)

    :returns:   The multi agent navground environment.
    """
    if reward is None:
        reward = NullReward()
    group = GroupConfig(indices=Indices(indices),
                        sensor=sensor,
                        reward=reward,
                        action=action,
                        observation=observation,
                        color=color,
                        tag=tag)
    return MultiAgentNavgroundEnv(
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


def make_shared_parallel_env_with_env(
    env: BaseEnv,
    max_number_of_agents: int | None = None,
    indices: IndicesLike = Indices.all()
) -> MultiAgentNavgroundEnv:
    """
    Creates a shared parallel environment
    using the configuration of the agent exposed
    in a (navground) single-agent environment.


    :param      env:                   The environment.
    :param      max_number_of_agents:  The maximal number of agents that we will expose.
        It needs to be specified only for scenarios
        that generate world with a variable number of agents.
    :param      indices:         The world indices of the agent to control.
        All other agents are controlled solely by navground.

    :raises TypeError: If ``env.unwrapped`` is not a subclass of :py:class:`.env.NavgroundEnv`

    """
    if not isinstance(env.unwrapped, NavgroundEnv):
        raise TypeError("Environment is not a NavgroundEnv")

    kwargs = dict(**env.unwrapped._spec)
    kwargs.pop('agent_index')
    kwargs.pop('reward')
    kwargs.pop('action')
    kwargs.pop('observation')
    kwargs.pop('sensor')
    kwargs['max_number_of_agents'] = max_number_of_agents
    group = list(env.unwrapped.groups_config)[0]
    group.indices = Indices(indices)
    return MultiAgentNavgroundEnv(groups=[group], **kwargs)
