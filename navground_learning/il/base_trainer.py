import pathlib
from typing import Any, Sequence
import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml
from imitation.util.logger import configure
from navground import core, sim
from navground_learning.behaviors.policy import PolicyBehavior
from navground_learning.utils import Expert
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common import torch_layers
import torch as th

from . import make_imitation_venv

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


class BaseTrainer:

    def __init__(self,
                 env: gym.Env | None = None,
                 parallel: bool = False,
                 n_envs: int = 1,
                 seed: int = 0,
                 verbose: bool = False,
                 log_directory: str = "logs",
                 log_formats: Sequence[str] = [],
                 net_arch: list[int] = [32, 32],
                 **kwargs: Any):

        timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
        if any(l != 'stdout' for l in log_formats):
            path = f"{log_directory}/{timestamp}"
        else:
            path = None
        self.logger = configure(path, log_formats)
        self.rng = np.random.default_rng(seed)
        self.env, self.config, self.original_behavior, self.sensor = make_imitation_venv(
            env=env, parallel=parallel, n_envs=n_envs, rng=self.rng, **kwargs)
        self.expert = Expert(self.config)
        if isinstance(self.config.observation_space, gym.spaces.Dict):
            extractor = torch_layers.CombinedExtractor
        else:
            extractor = torch_layers.FlattenExtractor
        self._policy = ActorCriticPolicy(
            observation_space=self.config.observation_space,
            action_space=self.config.action_space,
            lr_schedule=lambda _: th.finfo(th.float32).max,
            features_extractor_class=extractor,
            net_arch=net_arch)
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
