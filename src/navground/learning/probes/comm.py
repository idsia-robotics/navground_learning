from __future__ import annotations

import numpy as np
from navground import sim


class CommProbe(sim.RecordProbe):
    """
    A probe that records transmitted messages to a single dataset
    of shape ``(#agents, message size)``
    """

    dtype = np.float32

    def update(self, run: sim.ExperimentalRun) -> None:
        for agent in run.world.agents:
            if not agent.behavior:
                continue
            try:
                comm = agent.behavior._comm  # type: ignore[attr-defined]
            except AttributeError:
                continue
            self.data.append(comm)

    def get_shape(self, world: sim.World) -> list[int]:
        shapes = []
        for agent in world.agents:
            if not agent.behavior:
                continue
            try:
                shapes.append(agent.behavior.  # type: ignore[attr-defined]
                              _gym_agent.action_config.comm_space.shape)
            except AttributeError:
                continue
        assert len(set(shapes)) == 1
        return [len(shapes), *shapes[0]]
