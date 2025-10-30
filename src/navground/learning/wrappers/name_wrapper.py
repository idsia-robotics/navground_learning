from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from ..parallel_env import MultiAgentNavgroundEnv, BaseParallelEnv
from ..types import Action, Observation, Info

if TYPE_CHECKING:
    from gymnasium import Space

StepReturn = tuple[dict[str, Observation], dict[str, float], dict[str, bool],
                   dict[str, bool], dict[str, Info]]
ResetReturn = tuple[dict[str, Observation], dict[str, Info]]


class NameWrapper(BaseParallelWrapper[str, Observation, Action]):
    """
    This wrapper renames the agent using their *navground* tags
    as prefix "tag1-tag2-...-tagn" or "agent" (if no tag is set).

    Indices are appended to the predix, e.g.
    agent_0, agent_1, ... .

    :param env: The wrapped environment
    """

    def __init__(self, env: BaseParallelEnv) -> None:
        assert isinstance(env.unwrapped, MultiAgentNavgroundEnv)
        self.env = env  # type: ignore
        self._env = env
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
            i for name, i in self._original_names.items()
            if name.startswith(group)
        ]

    @property
    def agents(self) -> list[str]:
        return [self._names[i] for i in self._env.agents]

    @agents.setter
    def agents(self, values: list[str]) -> None:
        self._env.agents = [
            self._original_names[i] for i in values
            if i in self._original_names
        ]

    def observation_space(self, agent: str) -> Space[Any]:
        return self._env.observation_space(self._original_names[agent])

    def action_space(self, agent: str) -> Space[Any]:
        return self._env.action_space(self._original_names[agent])

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:

        _obs, _infos = self._env.reset(seed=seed, options=options)
        obs = {self._names[k]: v for k, v in _obs.items()}
        infos = {self._names[k]: v for k, v in _infos.items()}
        return obs, infos

    def step(self, action: dict[str, Action]) -> StepReturn:
        r_action = {self._original_names[k]: v for k, v in action.items()}
        _obs, _reward, _terminated, _truncated, _infos = self._env.step(
            r_action)
        obs = {self._names[k]: v for k, v in _obs.items()}
        reward = {self._names[k]: v for k, v in _reward.items()}
        terminated = {self._names[k]: v for k, v in _terminated.items()}
        truncated = {self._names[k]: v for k, v in _truncated.items()}
        infos = {self._names[k]: v for k, v in _infos.items()}
        return obs, reward, terminated, truncated, infos
