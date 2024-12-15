from __future__ import annotations

import numpy as np
from navground import core, sim


class AdvanceTask(sim.Task, name="Advance"):

    def __init__(self,
                 direction: core.Vector2 = core.unit(0),
                 target: float = 1.0) -> None:
        super().__init__()
        self.direction = direction / np.linalg.norm(direction)
        self.target = target
        self._advancement = 0.0

    def update(self, agent: sim.Agent, world: sim.World, time: float) -> None:
        self._advancement = np.dot(
            agent.pose.position - self._initial_position, self._direction)
        if self.done():
            agent.controller.stop()

    def prepare(self, agent: sim.Agent, world: sim.World) -> None:
        self._initial_position = np.array(agent.pose.position)
        agent.controller.follow_velocity(self.direction)

    def done(self) -> bool:
        return self._advancement >= self._target_advancement

    @property
    @sim.register(core.unit(0), "direction")
    def direction(self) -> core.Vector2:
        return self._direction

    @direction.setter
    def direction(self, value: core.Vector2) -> None:
        norm = np.linalg.norm(value)
        if norm > 0:
            self._direction = value / norm

    @property
    @sim.register(1.0, "target advancement")
    def target(self) -> float:
        return self._target_advancement

    @target.setter
    def target(self, value: float) -> None:
        self._target_advancement = max(0, value)
