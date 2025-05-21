from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

import numpy as np
from navground import sim

from ..indices import Indices, IndicesLike

Condition = Callable[[sim.Agent, sim.World], bool]


class SuccessProbe(sim.RecordProbe):
    """
    A probe that records successes statuses to a single dataset
    of shape ``(#agents, )``, where

    - -1 = unknown
    - 0 = failure
    - 1 = success

    :param      ds:         The dataset
    :param      indices:    The indices of the agents to record
    :param      is_success: Success condition
    :param      is_success: Failure condition
    """

    dtype = np.int8

    def __init__(self,
                 ds: sim.Dataset,
                 indices: IndicesLike = Indices.all(),
                 is_success: Condition | None = None,
                 is_failure: Condition | None = None):
        super().__init__(ds)
        self._indices = Indices(indices)
        self._is_success = is_success
        self._is_failure = is_failure

    def update(self, run: sim.ExperimentalRun) -> None:
        if self._is_success or self._is_failure:
            for agent in self._indices.sub_sequence(run.world.agents):
                if not hasattr(agent, '_success'):
                    if self._is_success and self._is_success(agent, run.world):
                        agent._success = True  # type: ignore[attr-defined]
                    if self._is_failure and self._is_failure(agent, run.world):
                        agent._success = False  # type: ignore[attr-defined]

    def finalize(self, run: sim.ExperimentalRun) -> None:
        for agent in self._indices.sub_sequence(run.world.agents):
            self.data.push(getattr(agent, '_success', -1))

    def get_shape(self, world: sim.World) -> list[int]:
        return [len(self._indices.sub_sequence(world.agents))]

    @classmethod
    def with_indices(
            cls,
            indices: IndicesLike) -> Callable[[sim.Dataset], SuccessProbe]:
        """
        Creates a probe factory to record success statuses.

        :param      indices:  The indices of the agents to record

        :returns:   A callable that can be added to runs or experiments using
            :py:meth:`navground.sim.ExperimentalRun.add_record_probe` or
            :py:meth:`navground.sim.Experiment.add_record_probe`.
        """
        probe = functools.partial(cls, indices=indices)
        return cast('Callable[[sim.Dataset], SuccessProbe]', probe)

    @classmethod
    def with_criteria(
        cls,
        is_success: Condition | None = None,
        is_failure: Condition | None = None
    ) -> Callable[[sim.Dataset], SuccessProbe]:
        probe = functools.partial(cls,
                                  is_success=is_success,
                                  is_failure=is_failure)
        return cast('Callable[[sim.Dataset], SuccessProbe]', probe)
