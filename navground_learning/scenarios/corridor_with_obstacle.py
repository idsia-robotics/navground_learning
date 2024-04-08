from navground import core, sim

# very simple worlds with
# - 1 agent that starts at (0, uniform) and wants to got forwards
# - two long horizontal walls at y=0, width
# - one obstacle at (length, uniform)
# sim finishes when agent reaches +2length


class CorridorWithObstacle(sim.Scenario,
                           name="CorridorWithObstacle"):  # type: ignore

    _length = 10.0
    _width = 1.0
    _min_radius = 0.1
    _max_radius = 0.5

    def __init__(self,
                 length: float = 10.0,
                 width: float = 1.0,
                 min_radius: float = 0.1,
                 max_radius: float = 0.5):
        super().__init__()
        self._length = length
        self._width = width
        self._min_radius = min_radius
        self._max_radius = max_radius

    def init_world(self, world: sim.World, seed: int | None = None) -> None:
        super().init_world(world, seed=seed)
        world.bounding_box = sim.BoundingBox(0, 2 * self._length, 0,
                                             self._width)
        world.add_wall(
            core.LineSegment((-self._length, 0), (2 * self._length, 0)))
        world.add_wall(
            core.LineSegment((-self._length, self._width),
                             (3 * self._length, self._width)))

        assert len(world.agents) == 1
        rng = world.random_generator
        agent = world.agents[0]
        r = agent.radius + agent.behavior.safety_margin
        agent.pose = core.Pose2((0, rng.uniform(r, self._width - r)), 0)
        agent.controller.follow_direction((1, 0))
        agent.twist = core.Twist2((agent.behavior.optimal_speed, 0), 0)
        agent.last_cmd = agent.twist

        def t(world):
            return agent.position[0] > 2 * self._length

        world.set_termination_condition(t)

        ro = rng.uniform(self._min_radius, self._max_radius)
        margin = ro + r
        if rng.integers(1):
            y = rng.uniform(ro, self._width - margin)
        else:
            y = rng.uniform(margin, self._width - ro)
        world.add_obstacle(core.Disc((self._length, y), ro))

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
    @sim.register(0.1, "max radius")
    def max_radius(self) -> float:
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = max(0, value)
