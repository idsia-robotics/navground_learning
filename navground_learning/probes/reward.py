import numpy as np
from navground import sim

from ..config import WorldConfig
from ..reward import Reward


class RewardProbe(sim.RecordProbe):
    dtype = np.float64

    def __init__(self, ds: sim.Dataset, config: WorldConfig):
        super().__init__(ds)
        self._config = config
        self._reward: dict[int, Reward] = {}

    def prepare(self, run: sim.ExperimentalRun) -> None:
        for index, agent in self._config.init_agents(run.world).items():
            if agent.reward is not None:
                self._reward[index] = agent.reward
        super()._prepare(run)

    def update(self, run: sim.ExperimentalRun) -> None:
        for i, reward in self._reward.items():
            value = reward(agent=run.world.agents[i],
                           world=run.world,
                           time_step=run.time_step)
            self.data.push(value)

    def get_shape(self, world: sim.World) -> tuple[int, ...]:
        return (len(self._reward), )
