from __future__ import annotations

import logging
import sys
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

import gymnasium as gym
import numpy as np
import yaml
from navground import core, sim

from ..config import ActionConfig, GroupConfig, ObservationConfig, StateConfig
from ..policies.info_predictor import InfoPolicy
from ..types import Action, Array, Bounds, Info, Observation
from .clock import SyncClock
from .group import Agent, GymAgent, create_agents_in_groups

StepReturn = tuple[dict[int, Observation], dict[int, float], dict[int, bool],
                   dict[int, bool], dict[int, Info]]
ResetReturn = tuple[dict[int, Observation], dict[int, Info]]


def make_scenario(
        value: sim.Scenario | str | dict[str, Any] | None
) -> sim.Scenario | None:
    if isinstance(value, dict):
        value = yaml.dump(value)
    if isinstance(value, str):
        return sim.load_scenario(value)
    return value


class NavgroundBaseEnv:
    """
    This class describes an environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML
                     representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and
                     then treated as a :py:class:`str`.

    :param groups: The configuration of groups of agents controlled by the environment.
                   All other agents are controlled solely by navground.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

    :param time_step: The simulation time step applied at every :py:meth:`_step`.

    :param max_duration: If positive, it will signal a truncation after this simulated time.

    :param terminate_if_idle: Whether to terminate when an agent is idle

    :param truncate_outside_bounds: Whether to truncate when an agent exit the bounds

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

    :param stuck_timeout: The time to wait before considering an agent as stuck and terminate it.

    :param wait: Whether to signal termination/truncation only when
                 all agents have terminated/truncated.

    :param truncate_fast: Whether to signal truncation for all agents
                          as soon as one agent truncates.
    :param state: An optional global state configuration

    :param include_action: Whether to include field "navground_action" in the info

    :param include_success: Whether to include field "is_success" in the info

    :param init_success: The default value of success (valid until a termination condition is met)

    :param intermediate_success: Whether to include "is_success" in the info at intermediate
                                 steps (vs only at termination).

    """

    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 30}
    MAX_SEED = 2**31 - 1
    action_key = 'navground_action'

    def __init__(self,
                 scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 groups: Collection[GroupConfig] = tuple(),
                 max_number_of_agents: int | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 terminate_if_idle: bool = True,
                 bounds: Bounds | None = None,
                 truncate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0,
                 stuck_timeout: float = -1,
                 wait: bool = False,
                 truncate_fast: bool = False,
                 include_action: bool = True,
                 include_success: bool = True,
                 init_success: bool | None = None,
                 intermediate_success: bool = False,
                 state: StateConfig | None = None) -> None:

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]  # type: ignore
        self.groups_config = groups
        self.time_step = time_step
        self.max_duration = max_duration
        self.terminate_if_idle = terminate_if_idle
        self.bounds = bounds
        self.truncate_outside_bounds = truncate_outside_bounds
        self.render_mode = render_mode
        self.render_kwargs = render_kwargs
        self.realtime_factor = realtime_factor
        self.stuck_timeout = stuck_timeout
        self.wait = wait
        self.truncate_fast = truncate_fast
        self.include_action = include_action
        self.include_success = include_success
        self.intermediate_success = intermediate_success
        self.init_success = init_success
        self._scenario: sim.Scenario | None = make_scenario(scenario)
        self.max_number_of_agents = max_number_of_agents
        self.state_config = state
        self._truncatated = False
        self._terminated = False
        self._spec: dict[str, Any]
        self._init_spec(scenario)
        self._init()

    def action_config(self, index: int) -> ActionConfig | None:
        """
        Gets the action configuration for a (possible) agent

        :param      index:  The agent index

        :returns:   The action configuration, or None if undefined.
        """
        if index in self._possible_agents:
            g = self._possible_agents[index].gym
            return cast('ActionConfig', g.action_config) if g else None
        return None

    def get_cmd_from_action(self, index: int, action: Action,
                            time_step: float) -> core.Twist2 | None:
        """
        Convert action to navground command for a (possible) agent

        :param      index:  The agent index
        :param      action:     The action
        :param      time_step:  The time step

        :returns:   A control command or None if no agent is configured
                    at the given index.
        """
        if index in self._possible_agents:
            g = self._possible_agents[index].gym
            return g.get_cmd_from_action(action, time_step) if g else None
        return None

    def observation_config(self, index: int) -> ObservationConfig | None:
        """
        Gets the observation configuration for a (possible) agent

        :param      index:  The agent index

        :returns:   The observation configuration, or None if undefined.
        """
        if index in self._possible_agents:
            g = self._possible_agents[index].gym
            return cast('ObservationConfig',
                        g.observation_config) if g else None
        return None

    def _init_spec(
            self,
            scenario: sim.Scenario | str | dict[str, Any] | None) -> None:
        self._spec = {
            'max_number_of_agents': self.max_number_of_agents,
            'groups': self.groups_config,
            'scenario': scenario,
            'time_step': self.time_step,
            'terminate_if_idle': self.terminate_if_idle,
            'bounds': self.bounds,
            'max_duration': self.max_duration,
            'truncate_outside_bounds': self.truncate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor,
            'stuck_timeout': self.stuck_timeout,
            'wait': self.wait,
            'truncate_fast': self.truncate_fast,
            'include_action': self.include_action,
            'include_success': self.include_success,
            'init_success': self.init_success,
            'intermediate_success': self.intermediate_success,
            'state': self.state_config
        }

    def _init(self) -> None:
        self._world: sim.World | None = None
        self._agents: dict[int, Agent] = {}
        self._termination: dict[int, bool] = {}
        self._truncation: dict[int, bool] = {}
        self._success: dict[int, bool] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_space: gym.spaces.Box | None = None
        if self._scenario is None:
            self._possible_agents: dict[int, Agent] = {}
        else:
            world = self._scenario.make_world()
            if self.state_config:
                self._state_space = self.state_config.get_space(world)
            self._possible_agents = {
                i: agent
                for i, agent in create_agents_in_groups(
                    world, self.groups_config,
                    self.max_number_of_agents).items() if agent.gym is not None
            }
        self._observation_space = {
            i: cast("GymAgent", agent.gym).observation_space
            for i, agent in self._possible_agents.items()
        }
        self._action_space = {
            i: cast("GymAgent", agent.gym).action_space
            for i, agent in self._possible_agents.items()
        }
        if self.render_mode == "human" and not self._loop:
            import asyncio

            from navground.sim.ui import WebUI

            self._loop = asyncio.get_event_loop()
            self._rt_clock: SyncClock | None = SyncClock(
                self.time_step, factor=self.realtime_factor)
            self._ui: WebUI | None = WebUI(port=8002)
            self._loop.run_until_complete(self._ui.prepare())
        else:
            self._loop = None
            self._ui = None
            self._rt_clock = None
        # self._observation_item_space = {
        #     i: cast(GymAgent, agent.gym).observation_item_space
        #     for i, agent in self._possible_agents.items()
        # }

    # def get_observation_item_space(self, index: int) -> gym.spaces.Dict | None:
    #     return self._observation_item_space.get(index)

    def update_agents(self) -> None:
        if not self._world:
            return
        self._agents.clear()
        for group in self.groups_config:
            indices = group.indices.as_set(len(self._world.agents))
            for i in indices:
                ng_agent = self._world.agents[i]
                self._agents[i] = self._possible_agents[i]
                self._agents[i].navground = ng_agent
                gym_agent = self._agents[i].gym
                if gym_agent:
                    gym_agent.init(ng_agent.behavior, self._agents[i].state)
        self._termination = {i: False for i in self._agents}
        self._truncation = {i: False for i in self._agents}
        self._success = {}

    @property
    def init_args(self) -> dict[str, Any]:
        """
        Returns the arguments used to initialize the environment

        :returns:   The initialization arguments.
        """
        return self._spec

    @property
    def asdict(self) -> dict[str, Any]:
        """
        A JSON-able representation of the instance

        :returns:  A JSON-able dict
        """
        rs = {
            'groups': [g.asdict for g in self.groups_config],
            'time_step': self.time_step,
            'max_duration': self.max_duration,
            'terminate_if_idle': self.terminate_if_idle,
            'truncate_outside_bounds': self.truncate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor,
            'stuck_timeout': self.stuck_timeout,
            'wait': self.wait,
            'truncate_fast': self.truncate_fast,
            'include_action': self.include_action,
            'include_success': self.include_success,
            'init_success': self.init_success,
            'intermediate_success': self.intermediate_success
        }
        if self.state_config:
            rs['state'] = self.state_config.asdict
        if self._scenario:
            rs['scenario'] = yaml.safe_load(sim.dump(self._scenario))
        if self.bounds:
            rs['bounds'] = [p.tolist() for p in self.bounds]
        rs['type'] = type(self).__name__
        return rs

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """
        Load the class from the JSON representation

        :param value:  A JSON-able dict
        :returns: An instance of the class
        """
        t = value.get('type', '')
        for subcls in cls.__subclasses__():
            if subcls.__name__ == t:
                cls = subcls
                break
        env = cls()
        env.groups_config = [
            GroupConfig.from_dict(v) for v in value.get('groups', [])
        ]
        env.time_step = value.get('time_step', 0.1)
        env.max_duration = value.get('max_duration', -1)
        env.terminate_if_idle = value.get('terminate_if_idle', True)
        env.truncate_outside_bounds = value.get('truncate_outside_bounds',
                                                False)
        env.render_mode = value.get('render_mode')
        env.render_kwargs = value.get('render_kwargs', {})
        env.realtime_factor = value.get('realtime_factor', 1)
        env.stuck_timeout = value.get('stuck_timeout', -1)
        env.wait = value.get('wait', False)
        env.truncate_fast = value.get('truncate_fast', False)
        env.include_action = value.get('include_action', None)
        env.init_success = value.get('init_success', None)
        env.include_success = value.get('include_success', True)
        env.intermediate_success = value.get('intermediate_success', True)
        if 'state' in value:
            env.state_config = StateConfig.from_dict(value['state'])
        else:
            env.state_config = None
        if 'scenario' in value:
            env._scenario = sim.load_scenario(yaml.safe_dump(
                value['scenario']))
        else:
            env._scenario = None
        env.bounds = None
        if 'bounds' in value:
            ps = value['bounds']
            assert len(ps) == 2
            p1, p2 = ps
            env.bounds = (np.asarray(p1), np.asarray(p2))
        env._init()
        env._init_spec(value.get('scenario'))
        return env

    @property
    def scenario(self) -> sim.Scenario | None:
        """
        The navground scenario, if set.
        """
        return self._scenario

    def _reset(self,
               seed: int | None = None,
               options: dict[str, Any] | None = None) -> ResetReturn:
        world = self._world
        if world:
            world._close()
        self._world = sim.World()
        if seed is None and world:
            self._world.copy_random_generator(world)
        if seed is not None:
            seed = seed & self.MAX_SEED
        if not self._scenario:
            return {}, {}
        self._scenario.init_world(self._world, seed=seed)
        self._scenario.apply_inits(self._world)
        self._world._prepare()
        # Update dry does change last_cmd. Let's cache it to restore it afterwards
        last_cmds = [a.last_cmd for a in self._world.agents]
        self._world.update_dry(self.time_step, advance_time=False)
        self.update_agents()
        self._truncated = False
        self._terminated = False
        obs = self.get_observations()
        if self.render_mode == "human" and self._ui and self._loop:
            self._loop.run_until_complete(
                self._ui.init(self._world, bounds=self.bounds))
        infos = self.get_infos()
        # ... restore last_cmd
        for a, cmd in zip(self._world.agents, last_cmds, strict=True):
            a.last_cmd = cmd
        return obs, infos

    def _step(self, actions: dict[int, Action]) -> StepReturn:
        assert self._world is not None
        try:
            for index, action in actions.items():
                if self.agent_is_active(index):
                    agent = self._agents[index]
                    if agent.navground and agent.gym:
                        cmd = agent.gym.get_cmd_from_action(
                            action, self.time_step)
                        if agent.navground.behavior:
                            agent.navground.last_cmd = agent.navground.behavior.feasible_twist(
                                cmd)
        except Exception as e:
            logging.warning(e)
            pass
        self._world.actuate(self.time_step)
        self._world.update_dry(self.time_step, advance_time=False)
        if self.render_mode == "human" and self._ui and self._rt_clock and self._loop:
            self._loop.run_until_complete(self._ui.update(self._world))
            self._rt_clock.tick()
        rewards = self.get_rewards()
        self.update_termination()
        self.update_truncation()
        return (self.get_observations(), rewards,
                self.get_termination(), self.get_truncation(),
                self.get_infos())

    def get_rewards(self) -> dict[int, float]:
        if self._world:
            return {
                index:
                agent.reward(agent.navground, self._world, self.time_step)
                if self.agent_is_active(index) else 0
                for index, agent in self._agents.items()
                if agent.navground and agent.reward
            }
        return {}

    def render(self) -> Any:
        if self.render_mode == "rgb_array" and self._world:
            from navground.sim.ui.render import image_for_world

            return image_for_world(self._world,
                                   bounds=self.bounds,
                                   **self.render_kwargs)

    def close(self) -> None:
        if self._ui and self._loop:
            self._loop.run_until_complete(self._ui.stop())
            self._ui = None

    def get_observations(self) -> dict[int, Observation]:
        if self._world:
            for agent in self._agents.values():
                agent.update_state(self._world)
        return {
            index: agent.gym.update_observation()
            for index, agent in self._agents.items() if agent.gym
        }

    # Not truncated and not terminated
    def agent_is_active(self, index: int) -> bool:
        return not self.has_terminated(index) and not self.has_truncated(index)

    def update_success(self) -> None:
        if not self._world:
            return
        for index, agent in self._agents.items():
            if not self._termination[index]:
                if agent.navground:
                    if agent.success_condition:
                        if agent.success_condition(agent.navground,
                                                   self._world):
                            if index not in self._success:
                                self._success[index] = True
                            if agent.terminate_on_success:
                                self.terminate(index)
                                continue
                    if agent.failure_condition:
                        if agent.failure_condition(agent.navground,
                                                   self._world):
                            if index not in self._success:
                                self._success[index] = False
                            if agent.terminate_on_failure:
                                self.terminate(index)
                                continue

    def terminate(self, index: int) -> None:
        self._termination[index] = True
        if all(self._termination.values()):
            self._terminated = True

    def has_terminated(self, index: int) -> bool:
        if self.wait:
            return self._terminated
        return self._termination.get(index, False)

    def get_termination(self) -> dict[int, bool]:
        if self.wait and not self._terminated:
            return {i: False for i in self._agents}
        return self._termination

    def truncate(self, index: int) -> None:
        self._truncation[index] = True
        self._truncated = True

    def get_truncation(self) -> dict[int, bool]:
        # if self.wait and not all(self._truncation.values()):
        #     return {i: False for i in self._truncation}
        if self.truncate_fast and self._truncated:
            return {i: True for i in self._agents}
        return self._truncation

    def has_truncated(self, index: int) -> bool:
        if self.truncate_fast:
            return self._truncated
        return self._truncation.get(index, False)

    def update_termination(self) -> None:
        if not self._world:
            return
        self.update_success()
        if self._world.should_terminate():
            self._terminated = True
            return
        time = self._world.time
        for index, agent in self._agents.items():
            if not self._termination[index]:
                if agent.navground:
                    if self.terminate_if_idle and agent.navground.idle:
                        self.terminate(index)
                        continue
                    if self.stuck_timeout > 0 and agent.navground.has_been_stuck_since(
                            time - self.stuck_timeout):
                        self.terminate(index)
                        continue

    def update_truncation(self) -> None:
        if self._world and self.max_duration > 0 and self.max_duration <= self._world.time:
            self._truncated = True
            self._truncation = {i: True for i in self._agents}
            return
        out = self.should_check_if_outside_bounds
        for index, agent in self._agents.items():
            if self._termination[index] or self._truncation[index]:
                continue
            if agent.navground:
                if out and self.is_outside_bounds(agent.navground.position):
                    self.truncate(index)

    @property
    def should_check_if_outside_bounds(self) -> bool:
        return self.truncate_outside_bounds and self.bounds is not None

    def is_outside_bounds(self, position: core.Vector2) -> bool:
        if self.bounds is not None:
            return any(position < self.bounds[0]) or any(
                position > self.bounds[1])
        return True

    def get_agent_infos(self, index: int, agent: Agent) -> dict[str, Action]:
        info = {}
        if self.include_action and agent.gym:
            if self.agent_is_active(index):
                info[self.action_key] = agent.gym.get_action(self.time_step)
            else:
                space = self._action_space[index]
                if isinstance(space, gym.spaces.Box):
                    info[self.action_key] = np.zeros_like(space.low)
        if self.include_success:
            if not self.agent_is_active(index) or self.intermediate_success:
                success = self._success.get(index, self.init_success)
                if success is not None:
                    info['is_success'] = success  # type: ignore
        return info

    def get_infos(self) -> dict[int, dict[str, Action]]:
        return {
            index: self.get_agent_infos(index, agent)
            for index, agent in self._agents.items() if agent.gym
        }

    def get_policy(self, index: int) -> InfoPolicy:
        """
        A policy that returns the action computed by the navground agent.

        :returns:   The policy.
        """
        return InfoPolicy(action_space=self._action_space[index],
                          observation_space=self._observation_space[index],
                          key=self.action_key)

    def get_state(self) -> Array | None:
        if self._world and self.state_config:
            return self.state_config.get_state(self._world)
        return None

    @property
    def state_space(self) -> gym.spaces.Box | None:
        return self._state_space

    @property
    def has_state(self) -> bool:
        return self.state_config is not None
