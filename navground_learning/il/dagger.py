import tempfile
from typing import Any

import gymnasium as gym
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

from .base_trainer import BaseTrainer

# def train(scenario: sim.Scenario,
#           sensor: sim.Sensor,
#           runs: int = 1000,
#           time_step: float = 0.1,
#           duration: float = 300.0,
#           epochs: int = 1,
#           gym_agent: GymAgent = GymAgent(stack=1, fix_orientation=True),
#           parallel: bool = False,
#           n_envs: int = 8):
#     rng = np.random.default_rng(0)
#     env = make_venv(scenario=scenario,
#                     sensor=sensor,
#                     time_step=time_step,
#                     duration=duration,
#                     gym_agent=gym_agent,
#                     parallel=parallel,
#                     n_envs=n_envs,
#                     rng=rng)
#     bc_trainer = bc.BC(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         rng=rng,
#     )
#     expert = gym_agent.configure(scenario, sensor).get_expert()
#     with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
#         dagger_trainer = SimpleDAggerTrainer(
#             venv=env,
#             scratch_dir=tmpdir,
#             expert_policy=expert,
#             bc_trainer=bc_trainer,
#             rng=rng,
#         )
#         dagger_trainer.train(runs,
#                              bc_train_kwargs={'n_epochs': epochs},
#                              rollout_round_min_episodes=10)

#     return dagger_trainer.policy


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
        self.bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=self.rng,
            custom_logger=self.logger,
            policy=self._policy,
            **self._bc_kwargs
        )
        self.trainer = SimpleDAggerTrainer(
            venv=self.env,
            scratch_dir=self.temp_dir,
            expert_policy=self.expert,
            bc_trainer=self.bc_trainer,
            rng=self.rng,
            custom_logger=self.logger,
            **self._dagger_kwargs
        )

    def __del__(self):
        self.temp_dir.cleanup()
