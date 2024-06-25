import tempfile
from typing import Any

import gymnasium as gym
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    A simplified interface to :py:class:`imitation.algorithms.dagger.SimpleDAggerTrainer`

    :param env: the [navground] enviroment.
    :param parallel: whether to parallize the env
    :param n_envs: the number of vectorized envs
    :param seed: the random seed
    :param verbose: whether to print training logs
    :param kwargs: Param passed to the [vectorized] env constructor
    """

    def __init__(self,
                 env: gym.Env | None = None,
                 parallel: bool = False,
                 n_envs: int = 1,
                 seed: int = 0,
                 verbose: bool = False,
                 bc_kwargs: dict[str, Any] = {},
                 dagger_kwargs: dict[str, Any] = {},
                 **kwargs: Any):
        self._bc_kwargs = bc_kwargs
        self._dagger_kwargs = dagger_kwargs
        super().__init__(env, parallel, n_envs, seed, **kwargs)

    def init_trainer(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="dagger")
        self.bc_trainer = bc.BC(observation_space=self.venv.observation_space,
                                action_space=self.venv.action_space,
                                rng=self.rng,
                                custom_logger=self.logger,
                                policy=self._policy,
                                **self._bc_kwargs)
        self.trainer = SimpleDAggerTrainer(venv=self.venv,
                                           scratch_dir=self.temp_dir,  # type: ignore
                                           expert_policy=self.expert,  # type: ignore
                                           bc_trainer=self.bc_trainer,
                                           rng=self.rng,
                                           custom_logger=self.logger,
                                           **self._dagger_kwargs)

    def __del__(self):
        self.temp_dir.cleanup()
