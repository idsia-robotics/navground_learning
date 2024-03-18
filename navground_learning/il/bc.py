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


# def train(scenario: sim.Scenario,
#           sensor: sim.Sensor,
#           runs: int = 60,
#           time_step: float = 0.1,
#           duration: float = 60.0,
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
#     expert = gym_agent.configure(scenario, sensor).get_expert()
#     transitions = sample_expert_transitions(env, expert, rng, runs)
#     bc_trainer = bc.BC(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         demonstrations=transitions,
#         rng=rng,
#     )
#     bc_trainer.train(n_epochs=epochs)
#     return bc_trainer.policy


class Trainer(BaseTrainer):
    """
    A simplified interface to :py:class:`imitation.algorithms.bc.BC`

    :param env: the [navground] enviroment.
    :param parallel: whether to parallize the env
    :param n_envs: the number of vectorized envs
    :param seed: the random seed
    :param verbose: whether to print training logs
    :param runs: how many runs to collect at init
    :param kwargs: Param passed to the [vectorized] env constructor
    """

    def __init__(self,
                 env: gym.Env | None = None,
                 parallel: bool = False,
                 n_envs: int = 1,
                 seed: int = 0,
                 verbose: bool = False,
                 runs: int = 0,
                 **kwargs: Any,
                 ):
        self.transitions = None
        super().__init__(env, parallel, n_envs, seed, **kwargs)
        if runs:
            self.collect_runs(runs)

    def init_trainer(self) -> None:
        self.trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=self.transitions,
            rng=self.rng,
            custom_logger=self.logger,
        )

    def collect_runs(self, runs: int) -> None:
        """
        Collect training runs

        :param      runs:  The number of runs
        """
        self.transitions = sample_expert_transitions(self.env, self.expert,
                                                     self.rng, runs)
        self.trainer.set_demonstrations(self.transitions)
