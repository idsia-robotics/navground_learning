from __future__ import annotations

import warnings
from typing import Any, cast

import gymnasium as gym
import numpy as np
import re

from ..types import Action, EpisodeStart, Observation, PathLike, State


def space_for_onnx_tensor(x: Any) -> gym.spaces.Box:
    shape = [i for i in x.shape if isinstance(i, int)]
    ts = re.findall(r"tensor\((.*)\)", x.type)
    dtype: type[np.floating[Any]] | type[np.integer[Any]]
    if not ts:
        warnings.warn(f"Unknown type {x.type}", stacklevel=1)
        dtype = np.float32
    else:
        if ts[0] == "float":
            dtype = np.float32
        elif ts[0] in x.type:
            dtype = np.float64
        else:
            dtype = np.dtype(ts[0]).type
    return gym.spaces.Box(shape=shape, low=-np.inf, high=np.inf, dtype=dtype)


class OnnxPolicy:
    """
    This class implements the :py:class:`navground.learning.types.PolicyPredictor`
    protocol. It loads an onnx model andd then calls
    :py:meth:`onnxruntime.InferenceSession.run` to perform inference.

    :param      path:  The path
    :type       path:  PathLike
    """

    def __init__(self, path: PathLike):
        import onnxruntime as ort  # type: ignore[import-untyped]

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_sess = ort.InferenceSession(path, options)
        self.input_dims = [len(x.shape) for x in self.ort_sess.get_inputs()]
        xs = self.ort_sess.get_inputs()
        ys = self.ort_sess.get_outputs()
        if len(xs) == 1 and xs[0].name == "observation":
            self._observation_space: gym.spaces.Box | gym.spaces.Dict = space_for_onnx_tensor(
                xs[0])
        else:
            self._observation_space = gym.spaces.Dict(
                {x.name: space_for_onnx_tensor(x)
                 for x in xs})
        # TODO(Jerome): why can it have more than one output?
        # assert len(ys) == 1, ys
        self._action_space = space_for_onnx_tensor(ys[0])

    # single obs -> single action, None
    # multiple obs -> multiple obs, None

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = True) -> tuple[Action, State | None]:
        """
        Perform one inference, calling :py:meth:`onnxruntime.InferenceSession.run`.

        The observation should respect the model specifics, also exposed
        by :py:attr:`observation_space`.

        :param      observation:    The input
        :param      state:          The state (ignored)
        :param      episode_start:  The episode start (ignored)
        :param      deterministic:  Whether deterministic (ignored)

        :returns:   The output of the inference and a `None` state
        """
        if not isinstance(observation, dict):
            vectorized = len(observation.shape) == self.input_dims[0]
            if not vectorized:
                observation = observation[np.newaxis, :]
            observation = {"observation": observation}
        else:
            vectorized = all(
                len(v.shape) == length for v, length in zip(
                    observation.values(), self.input_dims, strict=True))
            if not vectorized:
                observation = {
                    k: v[np.newaxis, :]
                    for k, v in observation.items()
                }
        action = self.ort_sess.run(None, observation)[0]
        if not vectorized:
            action = action[0]
        return cast(Action, action), None

    @property
    def observation_space(self) -> gym.spaces.Box | gym.spaces.Dict:
        """
        The observation space
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        """
        The action space
        """
        return self._action_space
