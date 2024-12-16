from __future__ import annotations

import dataclasses as dc
import warnings
from types import TracebackType
from typing import Any, cast

import gymnasium as gym
import numpy as np
from navground import core

from ..types import Array
from .base import ActionConfig, DataclassConfig


def to_py(value: Array) -> Any:
    try:
        return value.item()
    except ValueError:
        pass
    return value.tolist()


class TemporaryBehaviorParams:

    def __init__(self, behavior: core.Behavior, params: dict[str, Any]):
        super().__init__()
        self._behavior = behavior
        self._params = params
        self._old_params: dict[str, Any] = {}

    def __enter__(self) -> None:
        for k, v in self._params.items():
            self._old_params[k] = getattr(self._behavior, k)
            setattr(self._behavior, k, v)

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        for k, v in self._old_params.items():
            setattr(self._behavior, k, v)


def single_param_space(low: Any = 0,
                       high: Any = 1,
                       dtype: str = 'float',
                       discrete: bool = False,
                       normalized: bool = False) -> gym.Space[Any]:
    if discrete:
        return gym.spaces.Discrete(start=low, n=int(high - low))
    if normalized:
        low = -1
        high = 1
    box_type = np.dtype(dtype).type
    return gym.spaces.Box(low, high, dtype=box_type)


def param_space(params: dict[str, dict[str, Any]],
                normalized: bool = False) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        key:
        single_param_space(normalized=normalized, **value)
        for key, value in params.items()
    })


@dc.dataclass(repr=False)
class ModulationActionConfig(DataclassConfig, ActionConfig, register_name="Modulation"):
    """
    Configuration of the conversion between modulation actions
    and control commands.

    :param params: The parameters to modulate.

    Actions are values of parameters of a behavior.

    """

    params: dict[str, dict[str, Any]] = dc.field(default_factory=dict)
    """The parameters to modulate as a dictionary name -> space configuration"""

    def __post_init__(self) -> None:
        self.param_space = param_space(self.params, normalized=True)

    def is_configured(self, warn: bool = False) -> bool:
        if not self.params:
            warnings.warn("No parameters", stacklevel=1)
            return False
        return True

    def normalize(self, key: str, value: Any) -> Any:
        param = self.params[key]
        if 'discrete' in param:
            return value
        low = param['low']
        high = param['high']
        return np.clip(-1, 1, -1 + 2 * (value - low) / (high - low))

    def de_normalize(self, key: str, value: Any) -> Any:
        param = self.params[key]
        if 'discrete' in param:
            return value
        low = param['low']
        high = param['high']
        return low + (value + 1) / 2 * (high - low)

    @property
    def space(self) -> gym.Space[Any]:
        """
        The action space is a flattened :py:class:`gymnasium.spaces.Dict`
        with one entry per controlled parameter.
        """
        if self.param_space:
            return gym.spaces.flatten_space(self.param_space)
        else:
            return self.param_space

    def get_params_from_action(self, action: Array) -> dict[str, Any]:
        values: dict[str,
                     Array] = gym.spaces.unflatten(self.param_space, action)
        return {k: to_py(self.de_normalize(k, v)) for k, v in values.items()}

    def get_cmd_from_action(self, action: Array,
                            behavior: core.Behavior | None,
                            time_step: float) -> core.Twist2:
        assert behavior is not None
        params = self.get_params_from_action(action)
        with TemporaryBehaviorParams(behavior, params):
            cmd = behavior.compute_cmd(time_step)
        return cmd

    def get_action(self, behavior: core.Behavior, time_step: float) -> Array:
        params = {
            k: self.normalize(k, getattr(behavior, k))
            for k in self.param_space
        }
        return cast(Array, gym.spaces.flatten(self.param_space, params))

    def configure(self, behavior: core.Behavior) -> None:
        pass

    @property
    def asdict(self) -> dict[str, Any]:
        rs = dc.asdict(self)
        rs['type'] = self.__class__.__name__
        return rs
