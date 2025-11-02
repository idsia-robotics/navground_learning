from __future__ import annotations

import numpy as np
from navground import core, sim


class CommSensor(sim.Sensor, name="Comm", include_properties_of=['Sensor']):
    """
    A sensor that receive messages broadcasted by other agents.
    that it stores in field "comm".

    :param      binarize:  Whether to binarize the received message
    :param      size:      The message size
    :param      name:      The namespace
    """

    def __init__(self, binarize: bool = True, size: int = 1, name: str = ''):
        sim.Sensor.__init__(self, name=name)
        self._binarize = binarize
        self._size = size

    @property
    @sim.register(True, "Whether to binarize comm")
    def binarize(self) -> bool:
        return self._binarize

    @binarize.setter
    def binarize(self, value: bool) -> None:
        self._binarize = value

    @property
    @sim.register(1, "size")
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = max(0, value)

    def update(self, agent: sim.Agent, world: sim.World,
               state: core.EnvironmentState) -> None:
        if not isinstance(state, core.SensingState):
            return
        cs = []
        for other in world.agents:
            if other is not agent:
                if other.behavior and hasattr(other.behavior, "_comm"):
                    cs.append(other.behavior._comm)
                else:
                    cs.append(np.zeros(self.size))
        data = np.concatenate(cs, dtype=np.float32)
        if self._binarize:
            data = (data > 0).astype(dtype=int)
        self.get_or_init_buffer(state, "comm").data = data

    def get_description(self) -> dict[str, core.BufferDescription]:
        if self.binarize:
            desc = core.BufferDescription([self.size], np.dtype(np.int8), 0, 1)
        else:
            # CHANGED low
            desc = core.BufferDescription([self.size], np.float32, -1, 1)
        return {self.get_field_name("comm"): desc}
