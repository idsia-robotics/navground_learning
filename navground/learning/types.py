from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias

try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        pass
import abc

import numpy as np
from navground import sim

from .register import Registrable

Array: TypeAlias = np.typing.NDArray[Any]
Observation: TypeAlias = dict[str, Array] | Array
Action: TypeAlias = Array
State: TypeAlias = tuple[Array, ...]
EpisodeStart: TypeAlias = Array
Info: TypeAlias = list[dict[str, Array]] | dict[str, Array]
PathLike: TypeAlias = os.PathLike[str] | str

Bounds: TypeAlias = tuple[np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]]


class JSONAble(Protocol):

    @property
    def asdict(self) -> dict[str, Any]:
        """
        Return a JSON-able representation

        :returns:  A JSON-able dict
        """
        ...

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """
        Load the class from the JSON representation

        :param value:  A JSON-able dict
        :returns: An instance of the class
        """
        ...


class Reward(abc.ABC, Registrable):
    """
    The reward protocol is a callable to compute
    scalar rewards for individual agents.
    """

    @abc.abstractmethod
    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        """
        Compute the reward for an agent.

        :param      agent:      The agent
        :param      world:      The simulated world the agent belongs to
        :param      time_step:  The time step of the simulation

        :returns:   A scalar reward number
        """


class PolicyPredictor(Protocol):
    """
    This class describes the predictor protocol.

    Same as :py:type:`stable_baselines3.common.type_aliases.PolicyPredictor`,
    included here to be self-contained.
    """

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False) -> tuple[Action, State | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        ...


class PolicyPredictorWithInfo(Protocol):
    """
    Similar to :py:class:`PolicyPredictor` but :py:meth:`predict`
    accepts info dictionaries.
    """

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False,
                info: Info | None = None) -> tuple[Action, State | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :param info: Dictionaries with generic information that is not part of observation and state.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        ...


AnyPolicyPredictor: TypeAlias = PolicyPredictor | PolicyPredictorWithInfo


def accept_info(func: Callable[..., Any]) -> bool:
    """
    Check whether the callable accept an ``info`` argument.

    Can be used to distinguish :py:class:`PolicyPredictorWithInfo`
    from :py:class:`PolicyPredictor`:

    >>> policy = MyPolicy(...)
    >>> accept_info(policy.predict)
    >>> False

    :param      func:  The function to test
    :returns:   True if it has an argument named ``info``
    """
    sig = inspect.signature(func)
    return 'info' in sig.parameters
