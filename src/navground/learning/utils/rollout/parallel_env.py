from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, SupportsFloat, cast

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from pettingzoo.utils.env import ParallelEnv

from ...types import Action, AnyPolicyPredictor, Array, Observation
from .env import stack_dict


@dc.dataclass
class Rollout:
    observation: dict[int, Observation]
    action: dict[int, Action]
    reward: dict[int, Array]
    termination: dict[int, bool]
    truncation: dict[int, bool]
    info: dict[int, dict[str, Array]]


def rollout(env: ParallelEnv,
            max_steps: int = 1000,
            policy: AnyPolicyPredictor | None = None,
            deterministic: bool = True,
            seed: int | None = None,
            options: dict = {}) -> Rollout:
    obss: dict[int, list[Observation]] = {i: [] for i in env.possible_agents}
    infos: dict[int, list[dict[str, Array]]] = {
        i: []
        for i in env.possible_agents
    }
    rews: dict[int, list[SupportsFloat]] = {i: [] for i in env.possible_agents}
    acts: dict[int, list[Action]] = {i: [] for i in env.possible_agents}
    obs, info = env.reset(seed=seed, options=options)
    for i, v in obs.items():
        obss[i].append(v)
    for i, v in info.items():
        infos[i].append(v)
    for _ in range(max_steps):
        if policy:
            act = {
                i: policy.predict(o, deterministic=deterministic)[0]
                for i, o in obs.items()
            }
        else:
            act = {i: env.action_space(i).sample() for i in env.agents}
        for i, v in act.items():
            acts[i].append(v)
        obs, rew, term, trunc, info = env.step(act)
        for i, v in obs.items():
            obss[i].append(v)
        for i, v in rew.items():
            rews[i].append(v)
        for i, v in info.items():
            infos[i].append(v)
        if all(term.values()) or any(trunc.values()):
            break
    mobs: dict[int, Observation] = {}
    for i, aobs in obss.items():
        if not aobs:
            continue
        if isinstance(env.observation_space(i), gym.spaces.Dict):
            mobs[i] = stack_dict(cast('list[dict[str, Array]]', aobs))
        else:
            mobs[i] = np.asarray(aobs)
    minfo: dict[int, dict[str, Array]] = {}
    for i, ainfo in infos.items():
        if ainfo:
            minfo[i] = stack_dict(ainfo)
    mrew: dict[int, Array] = {}
    for i, arew in rews.items():
        if arew:
            mrew[i] = np.asarray(arew)
    mact: dict[int, Action] = {}
    for i, aact in acts.items():
        if aact:
            mact[i] = np.asarray(aact)
    return Rollout(observation=mobs,
                   reward=mrew,
                   termination=term,
                   truncation=trunc,
                   info=minfo,
                   action=mact)
