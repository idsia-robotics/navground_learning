import dataclasses as dc
import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import yaml
from navground import core, sim

from ..config import WorldConfig, to_list
from ..core import Agent, GymAgent, SyncClock

StepReturn = tuple[dict[int, Any], dict[int, float], dict[int, bool],
                   dict[int, bool], dict[int, dict[str, Any]]]

ResetReturn = tuple[dict[int, Any], dict[int, dict[str, Any]]]


def make_scenario(
        value: sim.Scenario | str | dict[str, Any] | None
) -> sim.Scenario | None:
    if isinstance(value, dict):
        value = yaml.dump(value)
    if isinstance(value, str):
        return cast(sim.Scenario | None, sim.load_scenario(value))
    return cast(sim.Scenario | None, value)


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
                 config: WorldConfig = WorldConfig(),
                 max_number_of_agents: int | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 terminate_outside_bounds: bool = True,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0) -> None:

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]  # type: ignore
        self.config = config
        self.time_step = time_step
        self.max_duration = max_duration
        self.bounds = bounds
        self.terminate_outside_bounds = terminate_outside_bounds
        self.render_mode = render_mode
        self.render_kwargs = render_kwargs
        self.realtime_factor = realtime_factor
        self._spec: dict[str, Any] = {
            'max_number_of_agents': max_number_of_agents,
            'config': config,
            'scenario': scenario,
            'time_step': time_step,
            'bounds': bounds,
            'max_duration': max_duration,
            'bounds': bounds,
            'terminate_outside_bounds': terminate_outside_bounds,
            'render_mode': render_mode,
            'render_kwargs': render_kwargs,
            'realtime_factor': realtime_factor
        }
        if self.render_mode == "human":
            import asyncio

            from navground.sim.ui import WebUI

            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_event_loop(
            )
            self._rt_clock: SyncClock | None = SyncClock(
                self.time_step, factor=realtime_factor)
            self._ui: WebUI | None = WebUI(port=8002)
            self._loop.run_until_complete(self._ui.prepare())
        else:
            self._loop = None
            self._ui = None
            self._rt_clock = None

        self._world: sim.World | None = None
        self._scenario: sim.Scenario | None = make_scenario(scenario)
        world = sim.World()
        if self._scenario:
            self._scenario.init_world(world)
        self._possible_agents = {
            i: agent
            for i, agent in self.config.init_agents(
                world, max_number_of_agents).items() if agent.gym is not None
        }
        self._agents: dict[int, Agent] = {}
        self._observation_space = {
            i: cast(GymAgent, agent.gym).observation_space
            for i, agent in self._possible_agents.items()
        }
        self._action_space = {
            i: cast(GymAgent, agent.gym).action_space
            for i, agent in self._possible_agents.items()
        }

    def update_agents(self):
        if not self._world:
            return
        self._agents.clear()
        for groups in self.config.groups:
            indices = to_list(groups.indices, self._world.agents)
            for i in indices:
                ng_agent = self._world.agents[i]
                self._agents[i] = self._possible_agents[i]
                self._agents[i].navground = ng_agent
                self._agents[i].gym.init(ng_agent.behavior,
                                         self._agents[i].state)

    @property
    def init_args(self) -> dict[str, Any]:
        """
        Returns the arguments used to initialize the environment

        :returns:   The initialization arguments.
        """
        return self._spec

    @property
    def asdict(self) -> dict[str, Any]:
        rs = {
            'config': dc.asdict(self.config),
            'time_step': self.time_step,
            'max_duration': self.max_duration,
            'terminate_outside_bounds': self.terminate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor
        }
        if self._scenario:
            rs['scenario'] = yaml.safe_load(sim.dump(self._scenario))
        if self.bounds:
            rs['bounds'] = [p.tolist() for p in self.bounds]
        return rs

    def _reset(self, seed=None, options=None) -> ResetReturn:
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
        for a, cmd in zip(self._world.agents, last_cmds):
            a.last_cmd = cmd
        return obs, infos

    def _step(self, actions: dict[int, np.ndarray]) -> StepReturn:
        assert self._world is not None
        try:
            for index, action in actions.items():
                if index in self._agents:
                    agent = self._agents[index]
                    if agent.navground and agent.gym:
                        cmd = agent.gym.get_cmd_from_action(
                            action, self.time_step)
                        agent.navground.last_cmd = agent.navground.behavior.feasible_twist(
                            cmd, core.Frame.absolute)
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

    def render(self):
        if self.render_mode == "rgb_array" and self._world:
            from navground.sim.ui.render import image_for_world

            return image_for_world(self._world,
                                   bounds=self.bounds,
                                   **self.render_kwargs)

    def close(self) -> None:
        if self._ui and self._loop:
            self._loop.run_until_complete(self._ui.stop())
            self._ui = None

    def get_observations(
            self) -> dict[int, dict[str, np.ndarray] | np.ndarray]:
        if self._world:
            for agent in self._agents.values():
                agent.update_state(self._world)
        return {
            index: agent.gym.update_observations()
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
             or agent.navground.has_been_stuck_since(1.0)
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

    def get_infos(self) -> dict[int, dict[str, np.ndarray]]:
        return {
            index: {
                'navground_action': agent.gym.get_action(self.time_step)
            }
            for index, agent in self._agents.items() if agent.gym
        }
