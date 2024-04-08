import numpy as np
from navground import core, sim


class ForwardScenario(sim.Scenario, name='Forward'):  # type: ignore

    _width = 2.0
    _length = 6.0
    _min_number_of_obstacles = 0
    _max_number_of_obstacles = 0
    _min_obstacle_radius = 0.1
    _max_obstacle_radius = 0.1
    _margin = 0.0
    _periodic_x = True
    _periodic_y = True

    def __init__(self):
        sim.Scenario.__init__(self)
        self._rng = None

    def init_world(self, world: sim.World, seed: int | None = None) -> None:
        sim.Scenario.init_world(self, world, seed)
        rng = world.random_generator
        # rng = np.random.default_rng(seed)
        world.bounding_box = sim.BoundingBox(0, self.length, -self.width * 0.5,
                                             self.width * 0.5)
        if self.periodic_x:
            world.set_lattice(0, (0, self.length))
        if self.periodic_y:
            world.set_lattice(1, (-self.width * 0.5, self.width))

        for i, agent in enumerate(world.agents):
            if i == 0:
                e = (1, 0)
                p = core.Pose2((0, 0), rng.uniform(high=2 * np.pi))
            else:
                alpha = rng.uniform(high=2 * np.pi)
                e = np.array((np.cos(alpha), np.sin(alpha)))
                p = core.Pose2((rng.uniform(low=0, high=self.length),
                                rng.uniform(low=-self.width * 0.5,
                                            high=self.width * 0.5)), alpha)
            agent.controller.follow_direction(e)
            agent.pose = p

        world.space_agents_apart(self.margin, with_safety_margin=True)
        margin = self.margin
        if world.agents:
            margin += 2 * max(
                agent.behavior.safety_margin if agent.behavior else 0 +
                agent.radius for agent in world.agents)

        if self._min_number_of_obstacles >= self._max_number_of_obstacles:
            number_of_obstacles = self._max_number_of_obstacles
        else:
            number_of_obstacles = rng.integers(
                low=self._min_number_of_obstacles,
                high=self._max_number_of_obstacles)
        if number_of_obstacles > 0:
            world.add_random_obstacles(number_of_obstacles,
                                       min_radius=self.min_obstacle_radius,
                                       max_radius=self.max_obstacle_radius,
                                       margin=margin,
                                       max_tries=10_000)

    @property
    @sim.register(6.0, "length of the area")
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = max(0, value)

    @property
    @sim.register(2.0, "width of the area")
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = max(0, value)

    @property
    @sim.register(0, "min number of obstacles")
    def min_number_of_obstacles(self) -> int:
        return self._min_number_of_obstacles

    @min_number_of_obstacles.setter
    def min_number_of_obstacles(self, value: int) -> None:
        self._min_number_of_obstacles = max(0, value)

    @property
    @sim.register(0, "max number of obstacles")
    def max_number_of_obstacles(self) -> int:
        return self._max_number_of_obstacles

    @max_number_of_obstacles.setter
    def max_number_of_obstacles(self, value: int) -> None:
        self._max_number_of_obstacles = max(0, value)

    @property
    @sim.register(0.1, "min obstacle radius")
    def min_obstacle_radius(self) -> float:
        return self._min_obstacle_radius

    @min_obstacle_radius.setter
    def min_obstacle_radius(self, value: float) -> None:
        self._min_obstacle_radius = max(0, value)

    @property
    @sim.register(0.1, "max obstacle radius")
    def max_obstacle_radius(self) -> float:
        return self._max_obstacle_radius

    @max_obstacle_radius.setter
    def max_obstacle_radius(self, value: float) -> None:
        self._max_obstacle_radius = max(0, value)

    @property
    @sim.register(0.1, "margin")
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        self._margin = max(0, value)

    @property
    @sim.register(True, "periodic_x")
    def periodic_x(self) -> bool:
        return self._periodic_x

    @periodic_x.setter
    def periodic_x(self, value: bool) -> None:
        self._periodic_x = value

    @property
    @sim.register(True, "periodic_x")
    def periodic_y(self) -> bool:
        return self._periodic_y

    @periodic_y.setter
    def periodic_y(self, value: bool) -> None:
        self._periodic_y = value
