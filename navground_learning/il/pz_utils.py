from typing import Any

import numpy as np
import pettingzoo as pz
import supersuit
from imitation.data import types
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3.common.vec_env import VecEnv

from ..env.pz import MultiAgentNavgroundEnv


class RolloutInfoWrapper(BaseParallelWrapper):

    def __init__(self, env: pz.ParallelEnv):
        super().__init__(env)
        self._obs: dict[int, list[types.DictObs]] = {}
        self._rews: dict[int, list[float]] = {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        obss, infos = self.env.reset(seed=seed, options=options)
        self._obs = {
            i: [types.maybe_wrap_in_dictobs(obs)]
            for i, obs in obss.items()
        }
        self._rews = {i: [] for i in obss}
        return obss, infos

    def step(self, actions):
        obss, rews, terms, trucs, infos = self.env.step(actions)
        for i, obs in obss.items():
            self._obs[i].append(types.maybe_wrap_in_dictobs(obs))
        for i, rew in rews.items():
            self._rews[i].append(rew)

        for i, term in terms.items():
            done = term or trucs[i]
            if done:
                info = infos[i]
                assert "rollout" not in info
                info["rollout"] = {
                    "obs": types.stack_maybe_dictobs(self._obs[i]),
                    "rews": np.stack(self._rews[i]),
                }
        return obss, rews, terms, trucs, infos


def make_ma_venv(env: pz.ParallelEnv | None = None,
                 n_envs: int = 8,
                 processes: int = 1,
                 **kwargs: Any) -> tuple[VecEnv, MultiAgentNavgroundEnv]:

    if env is None:
        env = MultiAgentNavgroundEnv(**kwargs)
    if not isinstance(env.unwrapped, MultiAgentNavgroundEnv):
        raise TypeError("unwrapped env is not a MultiAgentNavgroundEnv")
    o_env = env.unwrapped
    env = RolloutInfoWrapper(env)
    penv = supersuit.vector.MarkovVectorEnv(env, black_death=True)
    # penv = supersuit.pettingzoo_env_to_vec_env_v1(env)
    penv = supersuit.concat_vec_envs_v1(penv,
                                        1,
                                        num_cpus=1,
                                        base_class="stable_baselines3")
    return penv, o_env
