from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

import gymnasium as gym
import numpy as np
from navground import core

from ..env.env import BaseEnv
from ..types import Action, Array, Info, Observation

if TYPE_CHECKING:
    from .env import BaseParallelEnv

T = TypeVar('T', bound=gym.Space[Any])


def stack_infos(values: dict[int, Info],
                all_of: set[str] = {'is_success'}) -> Info:
    if not values:
        return {}
    rs: dict[str, Any] = {}
    keys: set[str] = set([v for value in values.values() for v in value])
    keys -= all_of
    for k in all_of:
        ms = set(value[k] for value in values.values() if k in value)
        if any(m is False for m in ms):
            rs[k] = False
        elif ms and all(m is True for m in ms):
            rs[k] = True
    for k in keys:
        ns = [value[k] for value in values.values() if k in value]
        rs[k] = np.stack(ns)
    return rs


def stack_observations(
        values: dict[int, Observation],
        dtype: type[np.floating[Any]] = core.FloatType) -> Observation:
    if not values:
        return np.array([], dtype=dtype)
    value = next(iter(values.values()))
    if isinstance(value, Mapping):
        keys = value.keys()
        return {k: np.stack([v[k] for v in values.values()]) for k in keys}
    return np.stack(list(cast('dict[int, Array]', values).values()))


def stack_box_spaces(spaces: list[gym.spaces.Box]) -> gym.spaces.Box:
    low = np.stack([space.low for space in spaces])
    high = np.stack([space.high for space in spaces])
    return gym.spaces.Box(low, high)


def stack_observation_spaces(values: dict[int, T]) -> T:
    assert (values)
    value = next(iter(values.values()))
    if isinstance(value, Mapping):
        keys = value.keys()
        spaces = cast('Iterable[Mapping[str, gym.spaces.Box]]',
                      values.values())
        return gym.spaces.Dict({
            k:
            stack_box_spaces([space[k] for space in spaces])
            for k in keys
        })

    return cast(
        'T',
        stack_box_spaces(
            list(cast('dict[int, gym.spaces.Box]', values).values())))


def unstack_actions(values: Array, agents: list[int]) -> dict[int, Array]:
    return dict(zip(agents, values, strict=True))


def get_state_space(env: BaseParallelEnv) -> gym.spaces.Box | None:
    """
    Gets the state space, first by trying to read
    ``env.state_space`` directly and then from ``env.state``.

    :param env:   The environment

    :returns:   The state space or None if not defined.
    """

    try:
        return env.state_space  # type: ignore
    except AttributeError:
        pass
    env.reset()
    state = env.state()
    return gym.spaces.Box(low=-np.inf,
                          high=np.inf,
                          shape=state.shape,
                          dtype=state.dtype.type)


class JointEnv(gym.Env[Observation, Action]):
    """
    Wraps a multi-agent parallel environment as a single
    agent environment, stacking observation but aggregating
    rewards, terminations and truncations.

    Requires that *homogeneous* observations spaces (if ``state=False``) and
    action spaces.

    :param env:   The parallel environment
    :param state: Whether to return the global state as observations
                  (vs the stacked observation).
    """

    def __init__(self, env: BaseParallelEnv, state: bool = False) -> None:
        self.env = env
        self.use_state = False
        if state:
            self.use_state = True
            state_space = get_state_space(env)
            if state_space:
                self.observation_space = state_space
            else:
                raise ValueError("Cannot get state space")
        else:
            self.use_state = False
            observation_spaces = {
                i: env.observation_space(i)
                for i in env.possible_agents
            }
            self.observation_space = stack_observation_spaces(
                observation_spaces)
        action_spaces = cast(
            'list[gym.spaces.Box]',
            [env.action_space(i) for i in env.possible_agents])
        self.action_space = stack_box_spaces(action_spaces)

    @property
    def unwrapped(self) -> BaseEnv:
        return self.env.unwrapped  # type: ignore

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """
        Conforms to :py:meth:`gymnasium.Env.reset`.

        Resets the parallel environment and return stacked
        observation (or the global state, if :py:attr:`use_state` is true)
        and infos.
        """
        obs, infos = self.env.reset(seed=seed, options=options)
        if self.use_state:
            state: Observation | None = self.env.state()
            if not state:
                raise RuntimeError("Environment has no state")
            vobs = state
        else:
            vobs = stack_observations(obs)
        return vobs, stack_infos(infos)

    def step(self,
             action: Action) -> tuple[Observation, float, bool, bool, Info]:
        """
        Conforms to :py:meth:`gymnasium.Env.step`.

        Converts the action to an agent-indexed dictionary
        ``[act_0, act_1, ...]`` -> ``{0: act_0, 1: act_1, ...}``
        and forwards them to :py:meth:`pettingzoo.utils.env.ParallelEnv.step`.

        Returns stacked observations (or the global state, if :py:attr:`use_state` is true)
        and infos, and aggregated reward (sum), termination (all) , truncation (any).
        """
        obs, reward, terminated, truncated, infos = self.env.step(
            unstack_actions(action, self.env.agents))
        if self.use_state:
            state: Observation | None = self.env.state()
            if not state:
                raise RuntimeError("Environment has no state")
            vobs = state
        else:
            vobs = stack_observations(obs)
        return (vobs, sum(x for _, x in reward.items()),
                all(x for _, x in terminated.items()),
                any(x for _, x in truncated.items()), stack_infos(infos))
