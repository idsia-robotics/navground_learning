from __future__ import annotations

import datetime
import pathlib as pl
import warnings
from typing import TYPE_CHECKING, Any

try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        ...

import gymnasium as gym
import numpy as np

from ..types import PathLike
from .utils import maybe_make_venv

# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.logger import Logger, configure


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from imitation.util.logger import HierarchicalLogger
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import VecEnv

ExtractorFactory = Any

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


def make_policy(
        observation_space: gym.Space[Any],
        action_space: gym.Space[Any],
        net_arch: list[int] = [32, 32],
        features_extractor: ExtractorFactory | None = None,
        features_extractor_kwargs: dict[str, Any] = {}) -> ActorCriticPolicy:
    import torch
    from stable_baselines3.common import torch_layers
    from stable_baselines3.common.policies import ActorCriticPolicy

    if features_extractor is None:
        if isinstance(observation_space, gym.spaces.Dict):
            features_extractor = torch_layers.CombinedExtractor
        else:
            features_extractor = torch_layers.FlattenExtractor
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        features_extractor_class=features_extractor,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=net_arch)


def make_logger(log_directory: PathLike = "logs",
                stamp_log_directory: bool = True,
                log_formats: Sequence[str] = []) -> HierarchicalLogger:
    from imitation.util.logger import configure

    log_directory = pl.Path(log_directory)
    if any(fmt != 'stdout' for fmt in log_formats):
        if stamp_log_directory:
            timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
            log_directory = log_directory / timestamp
    return configure(log_directory, log_formats)


class BaseILAlgorithm:
    """
    The base class that wraps IL algorithms to implement a
    :py:class:`stable_baselines3.common.base_class.BaseAlgorithm`-like API.

    :param env: the environment.
    :param seed: the random seed
    :param policy: an optional policy
    :param policy_kwargs: or the kwargs to create it
    :param logger: an optional logger
    """

    def __init__(self,
                 env: gym.Env[Any, Any] | VecEnv | None = None,
                 seed: int = 0,
                 policy: Any = None,
                 policy_kwargs: Mapping[str, Any] = {'net_arch': [32, 32]},
                 logger: HierarchicalLogger | None = None):
        if logger is None:
            logger = make_logger()
        self._logger = logger
        self._trainer: Any = None
        self.rng = np.random.default_rng(seed)
        if env:
            self._env: VecEnv | None = maybe_make_venv(env=env, rng=self.rng)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
        else:
            self._env = None
        if policy:
            self._policy = policy
            if self._env:
                if (policy.action_space != self._env.action_space
                        or policy.observation_space
                        != self._env.observation_space):
                    raise ValueError("Policy and env have different spaces")
            else:
                self._observation_space = policy.observation_space
                self._action_space = policy.action_space
        else:
            if self._env:
                self._policy = make_policy(self._env.observation_space,
                                           self._env.action_space,
                                           **policy_kwargs)
            else:
                raise ValueError(
                    "At least one of env or policy must be not None")

    @property
    def action_space(self) -> gym.Space[Any]:
        """The action space"""
        return self._action_space

    @property
    def observation_space(self) -> gym.Space[Any]:
        """The observation space"""
        return self._observation_space

    @property
    def policy(self) -> Any:
        """
        Gets the policy
        """
        return self._policy

    @property
    def env(self) -> VecEnv | None:
        """The training env"""
        return self.get_env()

    @env.setter
    def env(self, env: gym.Env[Any, Any] | VecEnv) -> None:
        self.set_env(env)

    def set_logger(self, logger: HierarchicalLogger) -> None:
        """
        Sets the logger.

        :param      logger:  The logger
        """
        self._logger = logger
        if self._trainer:
            self._trainer.logger = logger

    def get_env(self) -> VecEnv | None:
        """
        Gets the training environment.

        :returns:   The environment.
        """
        return self._env

    def set_env(self, env: gym.Env[Any, Any] | VecEnv) -> None:
        """
        Sets the training environment.

        Rejects the enviroment if not compatible with the policy.

        :param  env:  The new environment
        """
        if (self.policy.action_space == env.action_space
                and self.policy.observation_space == env.observation_space):
            self._env = maybe_make_venv(env)
        else:
            warnings.warn("Environment is not compatible with policy",
                          stacklevel=1)

    @property
    def logger(self) -> HierarchicalLogger:
        """
        The logger
        """
        return self._logger

    @logger.setter
    def logger(self, logger: HierarchicalLogger) -> None:
        self.set_logger(logger)

    def learn(
        self,
        *args: Any,
        # callback: Callable | BaseCallback | list[BaseCallback] | None = None,
        **kwargs: Any
    ) -> Self:
        """
        Learns the policy, passing the arguments to the wrapped ``imitation`` trainer.
        like

        .. code-block:: python

           trainer.train(*args, **kwargs)

        :returns: self
        """

        # if callback:
        #     callback.init_callback_for_imitation(self)
        #     kwargs['on_batch_end'] = lambda steps: callback.step(steps=steps)
        if self._trainer:
            self._trainer.train(*args, **kwargs)
        else:
            warnings.warn("Missing trainer", stacklevel=1)
        return self

    def save(self, path: pl.Path | str) -> None:
        """
        Saves the model using :py:func:`stable_baselines3.common.save_util.save_to_zip_file`

        :param      path:  The path to the directory where to create
        """
        from stable_baselines3.common.save_util import save_to_zip_file

        data = {'policy_kwargs': self.policy._get_constructor_parameters()}
        save_to_zip_file(save_path=path,
                         data=data,
                         params={'policy': self.policy.state_dict()},
                         pytorch_variables={})

    @classmethod
    def load(cls, path: pl.Path | str, env: gym.Env[Any, Any] | VecEnv | None = None) -> Self:
        """
        Loads a model using :py:func:`stable_baselines3.common.save_util.load_from_zip_file`

        :param      path:  The path to the saved model
        :param      env:   An optional training environment

        :returns:   the loaded algorithm
        """
        import torch
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.save_util import load_from_zip_file

        torch.serialization.add_safe_globals([ActorCriticPolicy])

        data, params, pytorch_variables = load_from_zip_file(path)
        if data:
            policy_kwargs: dict[str, Any] = data.get('policy_kwargs', {})
        else:
            policy_kwargs = {}
        policy = ActorCriticPolicy(**policy_kwargs)
        policy.load_state_dict(params['policy'])  # type: ignore[arg-type]
        return cls(env=env, policy=policy)
