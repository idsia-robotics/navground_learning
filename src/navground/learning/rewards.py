from __future__ import annotations

import dataclasses as dc
from collections.abc import Mapping
from typing import Any, cast

try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        ...

import numpy as np
from navground import core, sim

from .types import Reward


class NullReward(Reward, register_name="Zero"):
    """
        A dummy reward that returns always zero
    """

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


class EfficacyReward(Reward, register_name="Efficacy"):
    """
        A reward that returns the agent's
        :py:attr:`navground.core.Behavior.efficacy`

    """

    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        if agent.behavior:
            return agent.behavior.efficacy
        return 0


@dc.dataclass
class SocialReward(Reward, register_name="Social"):
    """
    Reward function for social navigation, inspired by [TODO add citation]

    It returns a weighted sum of

    - violations of the social margin ([-1, 0], weight ``alpha``)
    - violations of the safety margin ([-1, 0], weight ``beta``)
    - efficacy ([-1, 0], weight 1)

    so that it is lower or equal to zero, which corresponds to no violations,
    while moving at optimal speed towards the target.

    :param      alpha:                  The weight of social margin violations
    :param      beta:                   The weight of safety violations
    :param      critical_safety_margin: Violation of this margin has maximal penalty of -1
    :param      safety_margin:          Violations between this and the critical
                                        safety_margin have a linear penalty. If not set,
                                        it defaults to the agent's own safety_margin.
    :param      beta:                   The weight of safety violation
    :param      default_social_margin:  The default social margin
    :param      social_margins:         The social margins assigned to neighbors' ids

    :returns:   A function that returns -1 if the safety margin is violated
                or weighted sum of social margin violations and efficacy.
    """
    alpha: float = 0.0
    """The weight of social margin violations"""
    beta: float = 1.0
    """The weight of safety margin violations"""
    critical_safety_margin: float = 0.0
    """
       The value above which we assign a maximal penality
       to safety margin violations
    """
    safety_margin: float | None = None
    """An optional value. If not set, it will use the behavior's safety margin"""
    default_social_margin: float = 0.0
    """The default value of the social margin"""
    social_margins: dict[int, float] = dc.field(default_factory=dict)
    """The social margins to be applied to specific type of neighbors"""

    def __post_init__(self) -> None:
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

    def get_dict(self) -> dict[str, Any]:
        return dc.asdict(self)

    @classmethod
    def make_from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(**value)
