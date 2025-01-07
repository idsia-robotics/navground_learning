from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .base import BaseILAlgorithm

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gymnasium as gym
    import numpy
    from imitation.data.types import Transitions
    from imitation.util.logger import HierarchicalLogger
    from stable_baselines3.common.vec_env import VecEnv

    from . import rollout


def sample_expert_transitions(env: VecEnv,
                              expert: rollout.AnyPolicy,
                              rng: numpy.random.Generator,
                              runs: int = 1000) -> Transitions:
    from imitation.data.rollout import flatten_trajectories, make_sample_until

    from . import rollout

    rollouts = rollout.rollout(
        expert,
        env,
        make_sample_until(min_timesteps=None, min_episodes=runs),
        rng=rng,
    )
    return flatten_trajectories(rollouts)


class BC(BaseILAlgorithm):
    """
    A simplified interface to :py:class:`imitation.algorithms.bc.BC`

    :param env: the environment.
    :param seed: the random seed
    :param policy: an optional policy
    :param policy_kwargs: or the kwargs to create it
    :param logger: an optional logger
    :param runs: how many runs to collect at init
    :param expert: the expert to imitate.
                   If not set, it will default to the policy retrieved
                   from the env attribute "policy" (if set) else to `None`.
    :param bc_kwargs: parameters passed to the
        :py:class:`imitation.algorithms.bc.BC` constructor
    """

    def __init__(self,
                 env: gym.Env[Any, Any] | VecEnv | None = None,
                 seed: int = 0,
                 policy: Any = None,
                 policy_kwargs: Mapping[str, Any] = {'net_arch': [32, 32]},
                 logger: HierarchicalLogger | None = None,
                 runs: int = 0,
                 expert: rollout.AnyPolicy | None = None,
                 bc_kwargs: Mapping[str, Any] = {}):
        import imitation.data.rollout
        from imitation.algorithms import bc

        from . import rollout

        # TODO(Jerome): monkey patching for now
        # In the future, better to subclass BC.
        imitation.data.rollout.generate_trajectories = rollout.generate_trajectories

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
        self.transitions: Transitions | None = None
        self._bc_kwargs = bc_kwargs
        if runs:
            self.collect_runs(runs)
        self._trainer = bc.BC(
            observation_space=self.observation_space,
            action_space=self.action_space,
            demonstrations=self.transitions,
            rng=self.rng,
            policy=self._policy,
            **self._bc_kwargs,
        )
        self.set_logger(self.logger)

    def set_logger(self, logger: HierarchicalLogger) -> None:
        super().set_logger(logger)
        if self._trainer:
            self._trainer._bc_logger._logger = logger

    def collect_runs(self,
                     runs: int,
                     expert: rollout.AnyPolicy | None = None) -> None:
        """
        Collect training runs from an expert.

        :param runs:   The number of runs
        :param expert: the expert whose trajectories we want to collect.
                       If not set, it will default to the expert
                       configured at init.
        """
        if not expert:
            expert = self._expert
        if not expert:
            warnings.warn("No expert configured", stacklevel=1)
        if expert and self._trainer and self.env and runs > 0:
            self.transitions = sample_expert_transitions(
                self.env, expert, self.rng, runs)
            self._trainer.set_demonstrations(self.transitions)
