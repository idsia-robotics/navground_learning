import pathlib
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from imitation.util.logger import HierarchicalLogger
from navground import core, sim
from navground_learning.behaviors.policy import PolicyBehavior
from navground_learning.utils import get_expert
from stable_baselines3.common.logger import Logger

from . import make_imitation_venv


class BaseTrainer:

    def __init__(self,
                 env: gym.Env | None = None,
                 parallel: bool = False,
                 n_envs: int = 1,
                 seed: int = 0,
                 verbose: bool = False,
                 **kwargs: Any):

        self.verbose = verbose
        if not self.verbose:
            self.logger = HierarchicalLogger(Logger(None, []), [])
        else:
            self.logger = None
        self.rng = np.random.default_rng(seed)
        self.env, self.config, self.original_behavior, self.sensor = make_imitation_venv(
            env=env, parallel=parallel, n_envs=n_envs, rng=self.rng, **kwargs)
        self.expert = get_expert(self.config)
        self.init_trainer()
        self.behavior = PolicyBehavior.clone_behavior(self.original_behavior,
                                                      policy=self.policy,
                                                      config=self.config)

    def make_behavior(self,
                      behavior: core.Behavior | None = None) -> PolicyBehavior:
        """
        Construct a behavior from the trained policy.

        :param      behavior:  The behavior to replicate

        :returns:   The configured policy behavior.
        """
        return PolicyBehavior.clone_behavior(behavior
                                             or self.original_behavior,
                                             policy=self.policy,
                                             config=self.config)

    def init_trainer(self) -> None:
        ...

    def train(self, *args, **kwargs) -> None:
        """
        Train the policy, passing all arguments to the ``imitation`` trainer.
        """
        self.trainer.train(*args, **kwargs)

    @property
    def policy(self) -> Any:
        """
        Gets the trained policy
        """
        return self.trainer.policy

    def yaml(self) -> str:
        b = yaml.safe_load(sim.dump(self.behavior))
        s = yaml.safe_load(sim.dump(self.sensor))
        return yaml.dump({'behavior': b, 'state_estimation': s})

    def save(self, path: pathlib.Path) -> None:
        """
        Save model and config to a path.

        :param      path:  The path to the directory to create
        """
        path.mkdir(parents=True, exist_ok=False)
        policy_path = path / 'policy.th'
        torch.save(self.policy, policy_path)
        self.behavior.reset_policy_path('policy.th')
        with open(path / 'policy.yaml', 'w') as f:
            f.write(self.yaml())
