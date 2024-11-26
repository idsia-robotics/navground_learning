import datetime
import pathlib
from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pettingzoo as pz
import torch
import yaml
from imitation.algorithms import bc
from imitation.util.logger import configure
from navground import core, sim
from stable_baselines3.common import torch_layers
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from ..behaviors.policy import PolicyBehavior
# from ..config import get_sensor_as_dict
from ..core import Agent, ControlActionConfig, Expert
from .utils import make_venv

ExtractorFactory = Any

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"


class BaseTrainer:

    bc_trainer: bc.BC
    trainer: Any
    agent: Agent

    def __init__(self,
                 env: gym.Env | pz.ParallelEnv | None = None,
                 parallel: bool = False,
                 n_envs: int = 1,
                 seed: int = 0,
                 verbose: bool = False,
                 log_directory: str = "logs",
                 log_formats: Sequence[str] = [],
                 net_arch: list[int] = [32, 32],
                 features_extractor: ExtractorFactory | None = None,
                 features_extractor_kwargs: dict[str, Any] = {},
                 **kwargs: Any):

        timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
        if any(l != 'stdout' for l in log_formats):
            path = f"{log_directory}/{timestamp}"
        else:
            path = None
        self.logger = configure(path, log_formats)
        self.rng = np.random.default_rng(seed)
        self.venv, self.env = make_venv(env=env,
                                        parallel=parallel,
                                        n_envs=n_envs,
                                        rng=self.rng,
                                        **kwargs)
        self.agent = next(iter(self.env._possible_agents.values()))
        if not self.env._possible_agents:
            raise ValueError("No agent to imitate")
        if not (self.agent.navground and self.agent.gym):
            raise ValueError("Agent to imitate not valid")
        action_space = self.agent.gym.action_space
        observation_space = self.agent.gym.observation_space
        self.expert = Expert(action_space=action_space,
                             observation_space=observation_space)
        if features_extractor is None:
            if isinstance(observation_space, gym.spaces.Dict):
                features_extractor = torch_layers.CombinedExtractor
            else:
                features_extractor = torch_layers.FlattenExtractor
        self._policy = ActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            features_extractor_class=features_extractor,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=net_arch)
        self.init_trainer()
        if self.agent.navground.behavior:
            self.behavior = PolicyBehavior.clone_behavior(
                behavior=self.agent.navground.behavior,
                policy=self.policy,
                action_config=cast(ControlActionConfig,
                                   self.agent.gym.action_config),
                observation_config=self.agent.gym.observation_config)

    def make_behavior(self,
                      behavior: core.Behavior | None = None) -> PolicyBehavior:
        """
        Construct a behavior from the trained policy.

        :param      behavior:  The behavior to replicate

        :returns:   The configured policy behavior.
        """
        pb = self.behavior.clone()
        if behavior:
            pb.set_state_from(behavior)
        return pb

    def init_trainer(self) -> None:
        ...

    def train(self,
              *args,
              callback: BaseCallback | None = None,
              **kwargs) -> None:
        """
        Train the policy, passing all arguments to the ``imitation`` trainer.
        """
        # if callback:
        #     callback.init_callback_for_imitation(self)
        #     kwargs['on_batch_end'] = lambda steps: callback.step(steps=steps)
        self.trainer.train(*args, **kwargs)

    @property
    def policy(self) -> Any:
        """
        Gets the trained policy
        """
        return self.trainer.policy

    def yaml(self) -> str:
        rs = {'behavior': yaml.safe_load(sim.dump(self.behavior))}
        if self.agent.sensor:
            rs['sensor'] = yaml.safe_load(sim.dump(self.agent.sensor))
        return yaml.dump(rs)

    def save(self, path: pathlib.Path | str) -> None:
        """
        Save model and config to a path.

        :param      path:  The path to the directory to create
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=False)
        policy_path = path / 'policy.th'
        torch.save(self.policy, policy_path)
        self.behavior.reset_policy_path('policy.th')
        with open(path / 'policy.yaml', 'w') as f:
            f.write(self.yaml())

    def load(self, path: pathlib.Path | str) -> None:
        if isinstance(path, str):
            path = pathlib.Path(path)
        with open(path / 'policy.yaml', 'r') as f:
            data = yaml.safe_load(f)
        if 'behavior' in data:
            if 'policy_path' in data['behavior']:
                policy_path = data['behavior'].pop('policy_path')
            else:
                policy_path = 'policy.th'
            behavior = core.load_behavior(yaml.dump(data['behavior']))
            if behavior and isinstance(behavior, PolicyBehavior):
                behavior.policy_path = str(path / policy_path)
                self._policy = behavior._policy
                self.behavior = behavior
        else:
            self._policy = torch.load(path / 'policy.th')
        sensor: sim.Sensor | None = None
        if 'sensor' in data:
            se = sim.load_state_estimation(yaml.dump(data['sensor']))
            if isinstance(se, sim.Sensor):
                sensor = se
        self.agent = Agent(sensor=sensor)
