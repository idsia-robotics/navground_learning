from typing import Callable, cast

import numpy as np
from navground import sim

from ..behaviors.policy import PolicyBehavior
from ..config import Indices, WorldConfig, get_elements_at
from ..core import ControlActionConfig


def to_bounds(bb: sim.BoundingBox) -> tuple[np.ndarray, np.ndarray]:
    return bb.p1, bb.p2


def is_outside(p: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    return any(p < bounds[0]) or any(p > bounds[1])


def is_any_agents_outside(
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        agent_indices: Indices = [0]) -> Callable[[sim.World], bool]:

    def f(world: sim.World) -> bool:
        nonlocal bounds
        if bounds is None:
            bounds = to_bounds(world.bounding_box)
        return any(
            is_outside(agent.position, bounds)
            for agent in get_elements_at(agent_indices, world.agents))

    return f


class EvaluationScenario(sim.Scenario):

    def __init__(self,
                 scenario: sim.Scenario,
                 config: WorldConfig = WorldConfig(),
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 terminate_outside_bounds: bool = True,
                 deterministic: bool = True):
        sim.Scenario.__init__(self)
        self._scenario = scenario
        self._config = config
        self._bounds = bounds
        self._terminate_outside_bounds = terminate_outside_bounds
        self._deterministic = deterministic

    def init_world(self, world: sim.World, seed: int | None = None) -> None:
        self._scenario.init_world(world, seed=seed)

        agents = self._config.init_agents(world)

        if self._terminate_outside_bounds:
            tc = is_any_agents_outside(self._bounds, list(agents.keys()))
            world.set_termination_condition(tc)

        for agent in agents.values():
            ng_agent = agent.navground
            if ng_agent is None:
                continue
            if agent.policy is not None and agent.gym is not None and ng_agent.behavior:
                ng_agent.behavior = PolicyBehavior.clone_behavior(
                    behavior=ng_agent.behavior,
                    policy=agent.policy,
                    action_config=cast(ControlActionConfig,
                                       agent.gym.action_config),
                    observation_config=agent.gym.observation_config,
                    deterministic=self._deterministic)
                if agent.sensor is not None:
                    ng_agent.state_estimation = agent.sensor
