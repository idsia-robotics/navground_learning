from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np

from ..types import Array


def get_number_of_batches_in_array(value: Array, space: gym.spaces.Box) -> int:
    if len(value.shape) == len(space.shape):
        return 0
    if len(value.shape) == len(space.shape) + 1:
        return cast(int, value.shape[0])
    raise ValueError(f"Invalid shape {value.shape} for space {space}")


def get_number_of_batches_in_dict(value: dict[str, Array],
                                  space: gym.spaces.Dict) -> int:
    keys = set(space)
    batches = [
        get_number_of_batches_in_array(v, cast(gym.spaces.Box, space[k]))
        for k, v in value.items()
        if k in keys and isinstance(space[k], gym.spaces.Box)
    ]
    if not batches or len(set(batches)) > 1:
        raise ValueError(f"Invalid value {value} for space {space}")
    return batches[0]


def get_number_of_batches(value: Array | dict[str, Array],
                          space: gym.spaces.Space[Any]) -> int:
    if isinstance(space, gym.spaces.Dict):
        if isinstance(value, dict):
            return get_number_of_batches_in_dict(value, space)
        raise TypeError(f"Wrong type {type(value)} for Dict space")
    if isinstance(space, gym.spaces.Box):
        if isinstance(value, np.ndarray):
            return get_number_of_batches_in_array(value, space)
        raise TypeError(f"Wrong type {type(value)} for Box space")
    raise TypeError(f"Space of type {type(space)} not supported")
