from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

from .base import BaseILAlgorithm

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gymnasium as gym
    from imitation.util.logger import HierarchicalLogger
    from stable_baselines3.common.vec_env import VecEnv

    from .rollout import AnyPolicy


class DAgger(BaseILAlgorithm):
    """
    A simplified interface to out version of
    :py:class:`imitation.algorithms.dagger.SimpleDAggerTrainer`
    that accepts :py:type:`PolicyCallableWithInfo` experts.

    :param env: the environment.
    :param seed: the random seed
    :param policy: an optional policy
    :param policy_kwargs: or the kwargs to create it
    :param logger: an optional logger
    :param expert: the expert to imitate
    :param bc_kwargs: parameters passed to the
                      :imitation.algorithms.bc.BC: constructor
    :param dagger_kwargs: parameters passed to the
        :py:class:`imitation.algorithms.dagger.SimpleDAggerTrainer` constructor
    """

    def __init__(self,
                 env: gym.Env[Any, Any] | VecEnv | None = None,
                 seed: int = 0,
                 policy: Any = None,
                 policy_kwargs: Mapping[str, Any] = {'net_arch': [32, 32]},
                 logger: HierarchicalLogger | None = None,
                 expert: AnyPolicy | None = None,
                 bc_kwargs: Mapping[str, Any] = {},
                 dagger_kwargs: dict[str, Any] = {}):
        from imitation.algorithms import bc

        from .dagger_trainer import SimpleDAggerTrainer

        self._temp_dir: tempfile.TemporaryDirectory[Any] | None = None
        super().__init__(env=env,
                         seed=seed,
                         policy=policy,
                         policy_kwargs=policy_kwargs,
                         logger=logger)
        self._expert = expert
        if not expert and self.env:
            try:
                self._expert = self.env.get_attr("policy", [0])[0]
            except (AttributeError, IndexError):
                pass
        self._bc_kwargs = bc_kwargs
        self._dagger_kwargs = dagger_kwargs
        self._bc_trainer = bc.BC(observation_space=self.observation_space,
                                 action_space=self.action_space,
                                 rng=self.rng,
                                 policy=self._policy,
                                 **self._bc_kwargs)
        if self.env and self._expert:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="dagger")
            self._trainer = SimpleDAggerTrainer(
                venv=self.env,
                scratch_dir=self._temp_dir.name,
                expert_policy=self._expert,  # type: ignore[arg-type]
                bc_trainer=self._bc_trainer,
                rng=self.rng,
                **self._dagger_kwargs)
            self.set_logger(self.logger)

    def set_logger(self, logger: HierarchicalLogger) -> None:
        super().set_logger(logger)
        if self._bc_trainer:
            self._bc_trainer.logger = logger
            self._bc_trainer._bc_logger._logger = logger

    def __del__(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
