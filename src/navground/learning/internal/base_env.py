from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any, cast

try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        ...
import numpy as np
import yaml
from navground import core, sim

from ..config import GroupConfig
from ..policies.info_predictor import InfoPolicy
from ..types import Action, Bounds, Observation
from .clock import SyncClock
from .group import Agent, GymAgent, create_agents_in_groups

StepReturn = tuple[dict[int, Observation], dict[int, float], dict[int, bool],
                   dict[int, bool], dict[int, dict[str, Action]]]
ResetReturn = tuple[dict[int, Observation], dict[int, dict[str, Action]]]


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

    :param config: The configuration of the agents controlled by the environment.
                   All other agents are controlled solely by navground.

    :param max_number_of_agents: The maximal number of agents that we will expose.
                                 It needs to be specified only for scenarios
                                 that generate world with a variable number of agents.

    :param time_step: The simulation time step applied at every :py:meth:`_step`.

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
    action_key = 'navground_action'

    def __init__(self,
                 scenario: sim.Scenario | str | dict[str, Any] | None = None,
                 groups: Collection[GroupConfig] = tuple(),
                 max_number_of_agents: int | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: Bounds | None = None,
                 terminate_outside_bounds: bool = False,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0,
                 stuck_timeout: float = 1) -> None:

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]  # type: ignore
        self.groups_config = groups
        self.time_step = time_step
        self.max_duration = max_duration
        self.bounds = bounds
        self.terminate_outside_bounds = terminate_outside_bounds
        self.render_mode = render_mode
        self.render_kwargs = render_kwargs
        self.realtime_factor = realtime_factor
        self.stuck_timeout = stuck_timeout
        self._scenario: sim.Scenario | None = make_scenario(scenario)
        self.max_number_of_agents = max_number_of_agents
        self._spec: dict[str, Any]
        self._init_spec(scenario)
        self._init()

    def _init_spec(
        self, scenario: sim.Scenario | str | dict[str, Any] | None
    ) -> None:
        self._spec = {
            'max_number_of_agents': self.max_number_of_agents,
            'groups': self.groups_config,
            'scenario': scenario,
            'time_step': self.time_step,
            'bounds': self.bounds,
            'max_duration': self.max_duration,
            'terminate_outside_bounds': self.terminate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor,
            'stuck_timeout': self.stuck_timeout
        }

    def _init(self) -> None:
        self._world: sim.World | None = None
        self._agents: dict[int, Agent] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        if self._scenario is None:
            self._possible_agents: dict[int, Agent] = {}
        else:
            world = self._scenario.make_world()
            self._possible_agents = {
                i: agent
                for i, agent in create_agents_in_groups(
                    world, self.groups_config, self.max_number_of_agents).items()
                if agent.gym is not None
            }
        self._observation_space = {
            i: cast(GymAgent, agent.gym).observation_space
            for i, agent in self._possible_agents.items()
        }
        self._action_space = {
            i: cast(GymAgent, agent.gym).action_space
            for i, agent in self._possible_agents.items()
        }
        if self.render_mode == "human" and not self._loop:
            import asyncio

            from navground.sim.ui import WebUI

            self._loop = asyncio.get_event_loop(
            )
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
            'terminate_outside_bounds': self.terminate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor,
            'stuck_timeout': self.stuck_timeout
        }
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
        env.terminate_outside_bounds = value.get('terminate_outside_bounds',
                                                 False)
        env.render_mode = value.get('render_mode')
        env.render_kwargs = value.get('render_kwargs', {})
        env.realtime_factor = value.get('realtime_factor', 1)
        env.stuck_timeout = value.get('stuck_timeout', 1)
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
        self._world = sim.World()
        if seed is None and world:
            self._world.copy_random_generator(world)
        if seed is not None:
            seed = seed & self.MAX_SEED
        if not self._scenario:
            return {}, {}
        self._scenario.init_world(self._world, seed=seed)
        # Update dry does change last_cmd. Let's cache it to restore it afterwards
        last_cmds = [a.last_cmd for a in self._world.agents]
        self._world.update_dry(self.time_step, advance_time=False)
        self.update_agents()
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
                if index in self._agents:
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
        return (self.get_observations(), self.get_rewards(),
                self.has_terminated(), self.should_be_truncated(),
                self.get_infos())

    def get_rewards(self) -> dict[int, float]:
        if self._world:
            return {
                index: agent.reward(agent.navground, self._world,
                                    self.time_step)
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

    def has_terminated(self) -> dict[int, bool]:
        if self._world:
            should_terminate = self._world.should_terminate()
        else:
            should_terminate = False
        out = self.should_check_if_outside_bounds
        return {
            index:
            (should_terminate or agent.navground.idle
             or (self.stuck_timeout > 0
                 and agent.navground.has_been_stuck_since(self.stuck_timeout))
             or (out and self.is_outside_bounds(agent.navground.position)))
            for index, agent in self._agents.items() if agent.navground
        }

    @property
    def should_check_if_outside_bounds(self) -> bool:
        return self.terminate_outside_bounds and self.bounds is not None

    def is_outside_bounds(self, position: core.Vector2) -> bool:
        if self.bounds is not None:
            return any(position < self.bounds[0]) or any(
                position > self.bounds[1])
        return True

    def should_be_truncated(self) -> dict[int, bool]:
        value = (self._world is not None and self.max_duration > 0
                 and self.max_duration <= self._world.time)
        return {index: value for index in self._agents}

    def get_infos(self) -> dict[int, dict[str, Action]]:
        return {
            index: {
                self.action_key: agent.gym.get_action(self.time_step)
            }
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
