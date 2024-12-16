from __future__ import annotations

import numpy as np
from navground import core, sim
from ..tasks import AdvanceTask


class CorridorWithObstacles(sim.Scenario, name="CorridorWithObstacles"):

    def __init__(self,
                 number: int = 1,
                 length: float = 10.0,
                 width: float = 1.0,
                 min_radius: float = 0.1,
                 max_radius: float = 0.5):
        super().__init__()
        self._number = 1
        self._length = 10.0
        self._width = 1.0
        self._min_radius = min_radius
        self._max_radius = max_radius

    def init_world(self, world: sim.World, seed: int | None = None) -> None:
        super().init_world(world, seed)
        margin = 0.0
        world.add_wall(core.LineSegment((0, 0), (0, self.length)))
        world.add_wall(
            core.LineSegment((self.width, 0), (self.width, self.length)))
        for _ in range(self.number):
            p = None
            for _ in range(50):
                r = np.random.uniform(self._min_radius, self._max_radius)
                x = np.random.uniform(r + margin, self.width - r - margin)
                y = np.random.uniform(r, self.length - r)
                p = np.array((x, y))
                break
                if not any(
                        np.linalg.norm(p - o.disc.position) < (
                            r + o.disc.radius + margin)
                        for o in world.obstacles):
                    break
            if p is None:
                break
            world.add_obstacle(core.Disc(p, r))

        for agent in world.agents:
            agent.task = AdvanceTask(direction=np.array((0.0, 1.0)),
                                     target=self.length)
            x = np.random.uniform(agent.radius, self._width - agent.radius)
            theta = np.pi / 2
            agent.pose = core.Pose2((x, 0.0), theta)

    @property
    @sim.register(1, "number of obstacles")
    def number(self) -> int:
        return self._number

    @number.setter
    def number(self, value: int) -> None:
        self._number = max(0, value)

    @property
    @sim.register(10.0, "length of the corridor")
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = max(0, value)

    @property
    @sim.register(1.0, "width of the corridor")
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = max(0, value)

    @property
    @sim.register(1.0, "min radius")
    def min_radius(self) -> float:
        return self._min_radius

    @min_radius.setter
    def min_radius(self, value: float) -> None:
        self._min_radius = max(0, value)

    @property
    @sim.register(1.0, "max radius")
    def max_radius(self) -> float:
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = max(0, value)
