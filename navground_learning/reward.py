import dataclasses as dc
from typing import Any, Protocol, cast

import numpy as np
from navground import core, sim


class Reward(Protocol):

    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        ...

    @property
    def asdict(self) -> dict[str, Any]:
        ...


@dc.dataclass
class NullReward:

    type: str = dc.field(default="zero", init=False, repr=False)

    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        """
        A dummy zero reward.

        :param      agent:      The agent
        :param      world:      The world
        :param      time_step:  The time step

        :returns:   zero
        """
        return 0.0

    @property
    def asdict(self) -> dict[str, Any]:
        return dc.asdict(self)


@dc.dataclass
class SocialReward:

    alpha: float = 0.0
    beta: float = 1.0
    critical_safety_margin: float = 0.0
    safety_margin: float | None = None
    default_social_margin: float = 0.0
    social_margins: dict[int, float] = dc.field(default_factory=dict)
    type: str = dc.field(default="social", init=False, repr=False)
    """
    Reward function for social navigation, see (TODO add citation)

    :param      alpha:                  The weight of social margin violations
    :param      beta:                   The weight of safety violations
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

    def __post_init__(self):
        self._social_margin = core.SocialMargin(self.default_social_margin)
        for i, m in self.social_margins.items():
            self._social_margin.set(m, i)
        self._max_social_margin = self._social_margin.max_value

    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        if self.safety_margin is None:
            if agent.behavior:
                sm = agent.behavior.safety_margin
            else:
                sm = 0
        else:
            sm = self.safety_margin
        if sm > 0:
            max_violation = max(0, sm - self.critical_safety_margin)
            sv = world.compute_safety_violation(agent, sm)
            # if sv >= sm:
            #     return -1.0
            if sv == 0:
                r = 0.0
            elif sv > max_violation:
                r = -self.beta
            else:
                r = -self.beta * sv / max_violation
        else:
            r = 0
        if self._max_social_margin > 0:
            ns = world.get_neighbors(agent, self._max_social_margin)
            for n in ns:
                distance = cast(float,
                                np.linalg.norm(n.position - agent.position))
                margin = self._social_margin.get(n.id, distance)
                if margin > distance:
                    r += (distance - margin) * self.alpha * time_step
        if agent.task and agent.task.done():
            r += 1.0
        if agent.behavior:
            r += np.clip(0, 1, agent.behavior.efficacy) - 1
        return r

    @property
    def asdict(self) -> dict[str, Any]:
        return dc.asdict(self)
