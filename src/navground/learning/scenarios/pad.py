import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from navground import core, sim
from navground.sim.ui import svg_color


def make_thymio() -> sim.Agent:
    kinematics = core.kinematics.TwoWheelsDifferentialDriveKinematics(
        wheel_axis=0.094,
        max_speed=0.166,
        max_forward_speed=0.14,
        max_backward_speed=0.14)
    agent = sim.Agent(radius=0.1, kinematics=kinematics)
    agent.type = "thymio"
    return agent


def are_two_agents_on_the_pad(world: sim.World,
                              tolerance: float | None = None) -> bool:
    if tolerance is None:
        tolerance = world.pad_tolerance  # type: ignore[attr-defined]
    x = world.pad_width / 2 - tolerance  # type: ignore[attr-defined]
    return len([a for a in world.agents if abs(a.position[0]) < x]) > 1


def draw_pad(world: sim.World) -> str:
    return '<rect fill="red" width="0.5" height="0.6" x="-0.25" y="-0.3" fill-opacity="0.2"/>'


def _draw_tx_world(world: sim.World,
                   low: float = -1,
                   high: float = 1,
                   color_low: tuple[float, float, float] = (0, 0, 0),
                   color_high: tuple[float, float, float] = (0, 0, 1),
                   stroke: str = 'white',
                   binarize: bool = False) -> str:
    r = ''
    for agent in world.agents:
        b = agent.behavior
        if b and hasattr(b, '_comm'):

            tx = float(b._comm)  # type: ignore[attr-defined]
            if binarize:
                if tx > (high + low) / 2:
                    color = svg_color(*color_high)
                else:
                    color = svg_color(*color_low)
            else:
                x = (tx - low) / (high - low)
                rgb = np.asarray(color_low) * (1 -
                                               x) + np.asarray(color_high) * x
                color = svg_color(*rgb)
            r += (
                f'<circle cx="{agent.position[0]}" cy="{agent.position[1]}" '
                f'r="0.025" fill="{color}" stroke="{stroke}" stroke-width="0.01"/>'
            )
    return r


def draw_tx(low: float = -1,
            high: float = 1,
            color_low: tuple[float, float, float] = (0, 0, 0),
            color_high: tuple[float, float, float] = (0, 0, 1),
            stroke: str = 'white',
            binarize: bool = False) -> Callable[[sim.World], str]:

    return partial(_draw_tx_world,
                   low=low,
                   high=high,
                   color_low=color_low,
                   color_high=color_high,
                   stroke=stroke,
                   binarize=binarize)


def color_on_pad(entity: sim.Entity, world: sim.World) -> dict[str, str]:
    if isinstance(entity, sim.Agent):
        if are_two_agents_on_the_pad(world):
            return {'fill': 'red'}
    return {}


class PadScenario(sim.Scenario, name="Pad"):

    def __init__(self,
                 length: float = 2,
                 width: float = 0.6,
                 with_walls: bool = True,
                 start_min_x: float = -math.inf,
                 start_max_x: float = math.inf,
                 start_in_opposite_sides: bool = True,
                 pad_tolerance: float = 0.01,
                 pad_width: float = 0.5):
        super().__init__()
        self._length = length
        self._width = width
        self._with_walls = with_walls
        self._start_min_x = start_min_x
        self._start_max_x = start_max_x
        self._start_in_opposite_sides = start_in_opposite_sides
        self._pad_tolerance = pad_tolerance
        self._pad_width = pad_width

    @property
    @sim.register(1.0, "Length")
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = max(0, value)

    @property
    @sim.register(0.6, "Width")
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = max(0, value)

    @property
    @sim.register(True, "Whether to add walls")
    def with_walls(self) -> bool:
        return self._with_walls

    @with_walls.setter
    def with_walls(self, value: bool) -> None:
        self._with_walls = value

    @property
    @sim.register(-math.inf, "Minimal starting position")
    def start_min_x(self) -> float:
        return self._start_min_x

    @start_min_x.setter
    def start_min_x(self, value: float) -> None:
        self._start_min_x = max(0, value)

    @property
    @sim.register(math.inf, "Maximal starting position")
    def start_max_x(self) -> float:
        return self._start_max_x

    @start_max_x.setter
    def start_max_x(self, value: float) -> None:
        self._start_max_x = max(0, value)

    @property
    @sim.register(True, "Whether to start in opposing sides")
    def start_in_opposite_sides(self) -> bool:
        return self._start_in_opposite_sides

    @start_in_opposite_sides.setter
    def start_in_opposite_sides(self, value: bool) -> None:
        self._start_in_opposite_sides = value

    @property
    @sim.register(0.01, "Tolerance")
    def pad_tolerance(self) -> float:
        return self._pad_tolerance

    @pad_tolerance.setter
    def pad_tolerance(self, value: float) -> None:
        self._pad_tolerance = max(0, value)

    @property
    @sim.register(0.5, "Pad width")
    def pad_width(self) -> float:
        return self._pad_width

    @pad_width.setter
    def pad_width(self, value: float) -> None:
        self._pad_width = max(0, value)

    def init_world(self, world: sim.World, seed: int | None = None) -> None:
        super().init_world(world, seed=seed)
        rng = world.random_generator
        world.bounding_box = sim.BoundingBox(-self.length / 2, self.length / 2,
                                             -self.width / 2, self.width / 2)
        if self.with_walls:
            for i in (-1, 1):
                world.add_wall(
                    sim.Wall((-10 * self.length, i * self.width / 2),
                             (10 * self.length, i * self.width / 2)))
        while len(world.agents) < 2:
            world.add_agent(self.default_agent())

        def terminate(world: sim.World):
            return all(
                agent.behavior.target.direction.dot(
                    agent.behavior.position) > self.length / 2
                for agent in world.agents if agent.behavior
                and agent.behavior.target.direction is not None)

        world.set_termination_condition(terminate)

        assert len(world.agents) == 2
        sides = -1, 1
        orientations = 0, np.pi
        colors = 'gold', 'cyan'
        min_x = max(self.start_min_x, -self.length / 2)
        max_x = min(self.start_max_x, self.length / 2)
        if self.start_in_opposite_sides:
            min_x = max(min_x, self.pad_width / 2 + self.pad_tolerance)
        y = self.width / 4
        for agent, side, orientation, color in zip(world.agents,
                                                   sides,
                                                   orientations,
                                                   colors,
                                                   strict=True):
            if not agent.task:
                agent.task = sim.tasks.DirectionTask(direction=(-side, 0))
            if not agent.behavior:
                agent.behavior = core.behaviors.DummyBehavior()
            if not agent.kinematics:
                agent.kinematics = core.kinematics.TwoWheelsDifferentialDriveKinematics(
                    wheel_axis=0.94,
                    max_speed=0.166,
                    max_forward_speed=0.14,
                    max_backward_speed=0.14)
            if not agent.type:
                agent.type = "thymio"
            if not agent.radius:
                agent.radius = 0.1
            if not agent.color:
                agent.color = color
            agent.pose = core.Pose2(
                (side * rng.uniform(min_x, max_x), side * y), orientation)
        world.pad_width = self.pad_width  # type: ignore[attr-defined]
        world.pad_tolerance = self.pad_tolerance  # type: ignore[attr-defined]
        world.set("pad_width", self.pad_width)
        world.set("pad_tolerance", self.pad_tolerance)

    def default_agent(self) -> sim.Agent:
        return make_thymio()


def render_kwargs(comm: bool = False,
                  low: float = -1,
                  high: float = 1,
                  color_low: tuple[float, float, float] = (0, 0, 0),
                  color_high: tuple[float, float, float] = (0, 0, 1),
                  stroke: str = 'white',
                  binarize: bool = False) -> dict[str, Any]:
    """
    The kwargs to pass to the world renderer

    :param      comm: Whether to display transmitted messages using a LED.
    :param      low:  The message lower bound
    :param      high:  The message upper bound
    :param      color_low: The color associated to lower-valued messages
    :param      color_high: The color associated to higher-valued messages
    :param      stroke: The color of the LED boundary
    :param      binarize: Whether to binary message values.

    :returns:   The kwargs
    """
    rs = {'background_extras': [draw_pad], 'decorate': color_on_pad}
    if comm:
        rs['extras'] = [
            draw_tx(low=low,
                    high=high,
                    color_low=color_low,
                    color_high=color_high,
                    stroke=stroke,
                    binarize=binarize)
        ]

    return rs
