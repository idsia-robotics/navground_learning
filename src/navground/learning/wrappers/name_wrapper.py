from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pettingzoo.utils.wrappers import BaseParallelWrapper

from ..internal.base_env import ResetReturn, StepReturn
from ..parallel_env import MultiAgentNavgroundEnv
from ..types import Action

if TYPE_CHECKING:
    from pettingzoo.utils.env import ParallelEnv


class NameWrapper(BaseParallelWrapper):
    """
    This wrapper renames the agent using their *navground* tags
    as prefix "tag1-tag2-...-tagn" or "agent" (if no tag is set).

    Indices are appended to the predix, e.g.
    agent_0, agent_1, ... .

    :param env: The wrapped environment
    """

    def __init__(self, env: ParallelEnv) -> None:
        assert isinstance(env.unwrapped, MultiAgentNavgroundEnv)
        super().__init__(env)
        self._names: dict[int, str] = {
            i: f'{"-".join(agent.navground.tags) or "agent"}_{i}'
            for i, agent in env.unwrapped._possible_agents.items()
            if agent.navground
        }
        self._original_names: dict[str, int] = {
            v: k
            for k, v in self._names.items()
        }
        self.possible_agents = list(self._original_names)


    def get_indices(self, group: str) -> list[int]:
        """
        Gets the indices of agents with name <group>_<index>

        :param      group:  The group

        :returns:   The agent indices in the wrapped environment
        """
        return [
            i for name, i in self._original_names.items() if
            name.startswith(group)
        ]

    @property
    def agents(self) -> list[str]:
        return [self._names[i] for i in self.env.agents]

    @agents.setter
    def agents(self, values: list[str]) -> None:
        self.env.agents = [
            self._original_names[i] for i in values
            if i in self._original_names
        ]

    def observation_space(self, agent):
        return self.env.observation_space(self._original_names[agent])

    def action_space(self, agent):
        return self.env.action_space(self._original_names[agent])

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:

        obs, infos = self.env.reset(seed=seed, options=options)
        obs = {self._names[k]: v for k, v in obs.items()}
        infos = {self._names[k]: v for k, v in infos.items()}
        return obs, infos

    def step(self, action: dict[str, Action]) -> StepReturn:
        r_action = {self._original_names[k]: v for k, v in action.items()}
        obs, reward, terminated, truncated, infos = self.env.step(r_action)
        obs = {self._names[k]: v for k, v in obs.items()}
        reward = {self._names[k]: v for k, v in reward.items()}
        terminated = {self._names[k]: v for k, v in terminated.items()}
        truncated = {self._names[k]: v for k, v in truncated.items()}
        infos = {self._names[k]: v for k, v in infos.items()}
        return obs, reward, terminated, truncated, infos
