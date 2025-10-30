from __future__ import annotations

import dataclasses as dc
from collections.abc import Sequence
from functools import reduce
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np

from ...types import Action, AnyPolicyPredictor, Array, Observation, accept_info

# Could/should replicate the Torch RL Env.rolling
# that returns a dictionary of {'action', 'next', ...}


def stack_dict(ds: Sequence[dict[str, Array]]) -> dict[str, Array]:
    keys = reduce(lambda x, y: x | y, (set(d.keys()) for d in ds))
    return {key: np.array([d[key] for d in ds if key in d]) for key in keys}


@dc.dataclass
class Rollout:
    observation: Observation
    action: Action
    reward: Array
    termination: bool
    truncation: bool
    info: dict[str, Array]


def rollout(env: gym.Env[Any, Any],
            max_steps: int = 1000,
            policy: AnyPolicyPredictor | None | str = None,
            deterministic: bool = True,
            seed: int | None = None,
            options: dict[str, Any] | None = None) -> Rollout:
    pass_info = False
    if policy is not None and not isinstance(policy, str):
        pass_info = accept_info(policy.predict)
    obss: list[Observation] = []
    infos: list[dict[str, Array]] = []
    rews: list[SupportsFloat] = []
    acts: list[Action] = []
    obs, info = env.reset(seed=seed, options=options)
    for _ in range(max_steps):
        if isinstance(policy, str):
            act = info[policy]
        elif policy is None:
            act = env.action_space.sample()
        else:
            if pass_info:
                kwargs = {'info': info}
            else:
                kwargs = {}
            act, _ = policy.predict(obs, deterministic=deterministic, **kwargs)
        acts.append(act)
        obs, rew, term, trunc, info = env.step(act)
        obss.append(obs)
        rews.append(rew)
        infos.append(info)
        if term or trunc:
            break
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs = stack_dict(cast('list[dict[str, Array]]', obss))
    else:
        obs = np.asarray(obss)
    info = stack_dict(infos)
    return Rollout(observation=obs,
                   reward=np.asarray(rews),
                   termination=term,
                   truncation=trunc,
                   info=info,
                   action=np.asarray(acts))
