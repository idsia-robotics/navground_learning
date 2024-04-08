import logging
from typing import Any, Callable, Dict, Mapping, Tuple, cast

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.envs.registration import register
from navground import core, sim

from .utils import GymAgent, GymAgentConfig, SyncClock, get_behavior_and_sensor

# We consider:
# - one agent is imitated/learnt (at index `index`)
# - all the other agents do their own business

Reward = Callable[[sim.Agent, sim.World, float], float]


def null_reward(agent: sim.Agent, world: sim.World, time_step: float) -> float:
    """

    A dummy zero reward.

    :param      agent:      The agent
    :param      world:      The world
    :param      time_step:  The time step

    :returns:   zero
    """
    return 0.0


def social_reward(alpha: float = 0.0,
                  beta: float = 1.0,
                  critical_safety_margin: float = 0.0,
                  safety_margin: float | None = None,
                  default_social_margin: float = 0.0,
                  social_margins: Mapping[int, float] = {}) -> Reward:
    """
    Reward function for social navigation, see (TODO add citation)

    :param      alpha:                  The weight of social margin violations
    :param      beta:                   The weight of efficacy
    :param      critical_safety_margin: Violation of this margin has maximal penalty of -1
    :param      safety_margin:          Violations between this and the critical
                                        safety_margin have a linear penalty. If not set,
                                        it defaults to the agent's own safety_margin.
    :param      beta:                   The weight of efficacy
    :param      default_social_margin:  The default social margin
    :param      social_margins:         The social margins assigned to neighbors' ids

    :returns:   A function that returns -1 if the safety margin is violated
                or weighted sum of social margin violations and efficacy.
    """

    social_margin = core.SocialMargin(default_social_margin)
    for i, m in social_margins.items():
        social_margin.set(m, i)
    max_social_margin = social_margin.max_value

    def reward(agent: sim.Agent, world: sim.World, time_step: float) -> float:
        if safety_margin is None:
            if agent.behavior:
                sm = agent.behavior.safety_margin
            else:
                sm = 0
        else:
            sm = safety_margin
        max_violation = sm
        sv = world.compute_safety_violation(agent, sm)
        max_violation = sm - critical_safety_margin
        if sv >= sm:
            return -1.0
        elif sv > max_violation:
            r = -1.0
        else:
            r = -sv / max_violation
        if max_social_margin > 0:
            ns = world.get_neighbors(agent, max_social_margin)
            for n in ns:
                distance = cast(float,
                                np.linalg.norm(n.position - agent.position))
                margin = social_margin.get(n.id, distance)
                if margin > distance:
                    r += (distance - margin) * alpha * time_step
        if agent.task and agent.task.done():
            r += 1.0
        if agent.behavior:
            r += (agent.behavior.efficacy - 1) * beta  # * time_step
        return r

    return reward


class NavgroundEnv(gym.Env):
    """
    This class describes an environment that uses
    a :py:class:`navground.sim.Scenario` to
    generate and then simulate a :py:class:`navground.sim.World`.

    Actions and observations relates to a selected individual :py:class:`navground.sim.Agent`.

    The behavior is registered under the id ``"navground"``:

    >>> import gymnasium as gym
    >>> import navground_learning.env
    >>> from navground import sim
    >>>
    >>> scenario = sim.load_scenario(...)
    >>> env = gym.make("navground", scenario=scenario)


    :param scenario: The scenario to initialize all simulated worlds.
                     If a :py:class:`str`, it will be interpreted as the YAML representation of a scenario.
                     If a :py:class:`dict`, it will be dumped to YAML and then treated as a :py:class:`str`.

    :param agent_index: The world index of the selected agent, must be smaller than the number of agents.

    :param sensor: A sensor to produce observations for the selected agent.
                   If None, it will use the selected agent own state estimation, if a sensor.

    :param config: The configuration of the action and observation space to use. If ``None``, it will default to :py:class:`GymAgentConfig`.

    :param reward: The reward function to use. If ``None``, it will default to :py:func:`social_reward()`.

    :param time_step: The simulation time step applied at every :py:meth:`step`.

    :param max_duration: If positive, it will signal a truncation after this simulated time.

    :param bounds: The area to render and a fence for truncating processes when agents exit it.

    :param render_mode: The render mode.
                        If `"human"`, it renders a simulation in real time via
                        websockets (see :py:class:`navground.sim.ui.WebUI`).
                        If `"rgb_array"`, it uses :py:func:`navground.sim.ui.render.image_for_world`
                        to render the world on demand.

    :param render_kwargs: Arguments passed to :py:func:`navground.sim.ui.render.image_for_world`

    :param realtime_factor: a realtime factor for `render_mode="human"`: larger values speed up the simulation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 30}
    MAX_SEED = 2**31 - 1

    def __init__(self,
                 scenario: sim.Scenario | str | Dict[str, Any] | None = None,
                 agent_index: int = 0,
                 sensor: sim.Sensor | str | Dict[str, Any] | None = None,
                 config: GymAgentConfig | None = None,
                 reward: Reward | None = None,
                 time_step: float = 0.1,
                 max_duration: float = -1.0,
                 bounds: Tuple[np.ndarray, np.ndarray] | None = None,
                 truncate_outside_bounds: bool = True,
                 render_mode: str | None = None,
                 render_kwargs: Mapping[str, Any] = {},
                 realtime_factor: float = 1.0) -> None:

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]  # type: ignore

        self.scenario = scenario
        self.agent_index = agent_index
        self.sensor = sensor
        self.config = config
        self.reward = reward
        self.time_step = time_step
        self.bounds = bounds
        self.max_duration = max_duration
        self.bounds = bounds
        self.truncate_outside_bounds = truncate_outside_bounds
        self.render_mode = render_mode
        self.render_kwargs = render_kwargs
        self.realtime_factor = realtime_factor

        if reward is None:
            self._reward = social_reward()
        else:
            self._reward = reward

        if isinstance(scenario, dict):
            scenario = yaml.dump(scenario)
        if isinstance(scenario, str):
            self._scenario: sim._Scenario | None = sim.load_scenario(scenario)
        else:
            self._scenario = scenario

        if isinstance(sensor, dict):
            sensor = yaml.dump(sensor)
        if isinstance(sensor, str):
            se = sim.load_state_estimation(sensor)
            if isinstance(se, sim.Sensor):
                self._se: sim.Sensor | None = se
            else:
                print(f"State estimation {se} is not a sensor")
                self._se = None
        else:
            self._se = sensor
        if self._se:
            self._state: core.SensingState | None = core.SensingState()
            self._se.prepare(self._state)
        else:
            self._state = None

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

        if config is None:
            config = GymAgentConfig()

        self._config = config.configure(scenario=self._scenario,
                                        index=agent_index,
                                        sensor=self._se)
        self._world = None
        self.observation_space = self._config.observation_space
        self.action_space = self._config.action_space

    def get_behavior_and_sensor(
            self) -> Tuple[core.Behavior | None, sim.Sensor | None] | None:
        """
        Sample a navigation behavior from the scenario and possibly a sensor,
        if not configured to use a specific sensor

        :returns:   The agent navigation behavior and the sensor used by the MDP
        """
        if self._scenario is not None:
            return get_behavior_and_sensor(self._scenario, self.agent_index,
                                           self._se)
        return None

    def get_init_args(self) -> Dict[str, Any]:
        """
        Returns the arguments used to initialize the environment

        :returns:   The initialization arguments.
        """
        return {
            'scenario': self.scenario,
            'sensor': self.sensor,
            'config': self.config,
            'reward': self.reward,
            'time_step': self.time_step,
            'max_duration': self.max_duration,
            'bounds': self.bounds,
            'truncate_outside_bounds': self.truncate_outside_bounds,
            'render_mode': self.render_mode,
            'render_kwargs': self.render_kwargs,
            'realtime_factor': self.realtime_factor
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        world = self._world
        self._world = sim.World()
        if seed is None and world:
            self._world.copy_random_generator(world)
        if seed is not None:
            seed = seed & self.MAX_SEED
        if not self._scenario:
            return None, None
        self._scenario.init_world(self._world, seed=seed)
        # Update dry does change last_cmd. Let's cache it to restore it afterwards
        last_cmds = [a.last_cmd for a in self._world.agents]
        self._world.update_dry(self.time_step, advance_time=False)
        self._agent = self._world.agents[self.agent_index]
        self._agent.color = 'orange'
        behavior = self._agent.behavior
        state = behavior.environment_state
        if not self._state and isinstance(state, core.SensingState):
            self._state = state
        # self._gym_agent.use_behavior(behavior)
        # self._gym_agent.reset()
        self._gym_agent = GymAgent(self._config, behavior, self._state)
        obs = self._update_observations()
        if self.render_mode == "human" and self._ui:
            self._loop.run_until_complete(
                self._ui.init(self._world, bounds=self.bounds))
        infos = self._get_infos()
        # ... restore last_cmd
        for a, cmd in zip(self._world.agents, last_cmds):
            a.last_cmd = cmd
        return obs, infos

    def step(self, action: np.ndarray):
        assert self._world is not None
        try:
            cmd = self._gym_agent.get_cmd_from_action(action, self.time_step)
            self._agent.last_cmd = self._agent.behavior.feasible_twist(
                cmd, core.Frame.absolute)
        except Exception as e:
            logging.warning(e)
            pass
        self._world.actuate(self.time_step)
        self._world.update_dry(self.time_step, advance_time=False)
        obs = self._update_observations()
        # print(obs, self._gym_agent.action)

        if self.render_mode == "human" and self._ui:
            self._loop.run_until_complete(self._ui.update(self._world))
            self._rt_clock.tick()

        reward = self._reward(self._agent, self._world, self.time_step)
        return obs, reward, self._has_terminated(), self._should_be_truncated(
        ), self._get_infos()

    def render(self):
        if self.render_mode == "rgb_array":
            from navground.sim.ui.render import image_for_world

            return image_for_world(self._world,
                                   bounds=self.bounds,
                                   **self.render_kwargs)

    def close(self) -> None:
        super().close()
        if self._ui and self._loop:
            self._loop.run_until_complete(self._ui.stop())
            self._ui = None

    def _update_observations(self):
        if self._se:
            self._se.update(self._agent, self._world, self._state)
        return self._gym_agent.update_observations()

    def _has_terminated(self) -> bool:
        return (self._world.should_terminate() or self._agent.idle
                or self._agent.has_been_stuck_since(1.0))
        # if self._agent.task:
        #     agents_are_idle_or_stuck
        #     return self._agent.task.done()
        # return True

    def _should_be_truncated(self) -> bool:
        assert self._world is not None
        if self.max_duration > 0 and self.max_duration <= self._world.time:
            return True
        if self.truncate_outside_bounds and self.bounds is not None:
            p = self._agent.position
            return any(p < self.bounds[0]) or any(p > self.bounds[1])
        return False

    def _get_infos(self) -> Dict[str, np.ndarray]:
        # print(self._gym_agent.action.shape)
        return {'navground_action': self._gym_agent.get_action(self.time_step)}


register(
    id="navground",
    entry_point="navground_learning.env:NavgroundEnv",
)
