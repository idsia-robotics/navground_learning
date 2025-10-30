from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym

if TYPE_CHECKING:
    from torchrl.data import CompositeSpec  # type: ignore
    from torchrl.envs import EnvBase  # type: ignore
    from benchmarl.utils import DEVICE_TYPING  # type: ignore

from torchrl.data import Composite
from torchrl.envs.libs.pettingzoo import PettingZooWrapper  # type: ignore

from benchmarl.environments import TaskClass  # type: ignore

from ...parallel_env import MultiAgentNavgroundEnv
from ...wrappers.name_wrapper import NameWrapper


class NavgroundTaskClass(TaskClass):  # type: ignore[misc]

    def __init__(self,
                 *args: Any,
                 env: MultiAgentNavgroundEnv,
                 eval_env: MultiAgentNavgroundEnv | None = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not eval_env:
            eval_env = env
        self._env_spec = env._spec
        self._eval_env_spec = eval_env._spec
        self.config['env'] = env.asdict
        self._cont = all(
            isinstance(space, gym.spaces.Box)
            for space in env._action_space.values())
        if eval_env:
            self.config['eval_env'] = eval_env.asdict

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        config.pop('env')
        config.pop('eval_env', None)
        return lambda: PettingZooWrapper(
            NameWrapper(MultiAgentNavgroundEnv(**self._env_spec)),
            categorical_actions=not continuous_actions,
            device=device,
            seed=seed,
            return_state=self._env_spec.get('state', False),
            **config,
        )

    def get_eval_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        config.pop('env')
        config.pop('eval_env', None)
        return lambda: PettingZooWrapper(
            NameWrapper(MultiAgentNavgroundEnv(**self._eval_env_spec)),
            categorical_actions=not continuous_actions,
            device=device,
            seed=seed,
            return_state=self._env_spec.get('state', False),
            **config,
        )

    # def render_callback(self, *args, **kwargs):
    #     print('render')
    #     return super().render_callback(*args, **kwargs)

    def supports_continuous_actions(self) -> bool:
        return self._cont

    def supports_discrete_actions(self) -> bool:
        return not self._cont

    # def has_state(self) -> bool:
    #     return False

    def has_render(self, env: EnvBase) -> bool:
        return cast('str | None', env.render_mode) == 'rgb_array'

    def max_steps(self, env: EnvBase) -> int:
        return int(env._env.max_duration // env._env.time_step)

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        return cast('dict[str, list[str]]', env.group_map)

    def state_spec(self, env: EnvBase) -> CompositeSpec | None:
        if 'state' in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> CompositeSpec | None:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec:
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> CompositeSpec | None:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec:
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "navground"
