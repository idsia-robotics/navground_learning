from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from ..types import PathLike

if TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy


def export(policy: BasePolicy,
           path: PathLike,
           dynamic_batch_size: bool = True) -> None:
    """
    Export a policy (PyTorch) as onnx.

    :param      policy:              The policy
    :param      path:                Where to save the policy;
        by convention should have an ".onnx" suffix.
    :param      dynamic_batch_size:  Whether to enable dynamic batch size.
        If not set, it will export a model that evaluate a single sample.
    """
    import torch as th

    from .onnxable import OnnxablePolicy, OnnxablePolicyWithMultiInput

    # if hasattr(policy, 'actor'):
    #     policy = policy.actor
    obs = policy.observation_space.sample()
    if isinstance(obs, dict):
        dummy_input = tuple(
            th.from_numpy(v[np.newaxis, :]) for v in obs.values())
    else:
        dummy_input = (th.from_numpy(obs[np.newaxis, :]), )
    onnxable_model: th.nn.Module
    if isinstance(policy.observation_space, gym.spaces.Dict):
        keys = list(policy.observation_space.keys())
        onnxable_model = OnnxablePolicyWithMultiInput(policy, keys)
        input_names = keys
    else:
        onnxable_model = OnnxablePolicy(policy)
        input_names = ["observation"]

    if dynamic_batch_size:
        dynamic_axes: dict[str, dict[int, str]] | None = {
            k: {
                0: 'batch_size'
            }
            for k in input_names
        }
    else:
        dynamic_axes = None
    th.onnx.export(onnxable_model,
                   dummy_input,
                   path,
                   opset_version=17,
                   input_names=input_names,
                   dynamic_axes=dynamic_axes,
                   output_names=["action"])
