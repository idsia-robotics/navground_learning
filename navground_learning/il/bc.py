from typing import Any

import gymnasium as gym
from imitation.algorithms import bc
from imitation.data import rollout

from .base_trainer import BaseTrainer


def sample_expert_transitions(env, expert, rng, runs=1000):
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=runs),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


class Trainer(BaseTrainer):
    """
    A simplified interface to :py:class:`imitation.algorithms.bc.BC`

    :param env: the [navground] environment.
    :param parallel: whether to parallize the env
    :param n_envs: the number of vectorized envs
    :param seed: the random seed
    :param verbose: whether to print training logs
    :param runs: how many runs to collect at init
    :param kwargs: Param passed to the [vectorized] env constructor
    """

    def __init__(
        self,
        env: gym.Env | None = None,
        parallel: bool = False,
        n_envs: int = 1,
        seed: int = 0,
        verbose: bool = False,
        runs: int = 0,
        bc_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ):
        self.transitions = None
        self._bc_kwargs = bc_kwargs
        super().__init__(env, parallel, n_envs, seed, **kwargs)
        if runs:
            self.collect_runs(runs)

    def init_trainer(self) -> None:
        self.trainer = bc.BC(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            demonstrations=self.transitions,
            rng=self.rng,
            custom_logger=self.logger,
            policy=self._policy,
            **self._bc_kwargs,
        )
        self.bc_trainer = self.trainer

    def collect_runs(self, runs: int) -> None:
        """
        Collect training runs

        :param      runs:  The number of runs
        """
        self.transitions = sample_expert_transitions(self.venv, self.expert,
                                                     self.rng, runs)
        self.trainer.set_demonstrations(self.transitions)
