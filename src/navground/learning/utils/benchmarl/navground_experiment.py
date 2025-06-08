from __future__ import annotations

# import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarl.models.common import ModelConfig  # type: ignore[import-not-found]
    from benchmarl.algorithms.common import AlgorithmConfig  # type: ignore[import-not-found]
    from benchmarl.experiment import Callback, ExperimentConfig  # type: ignore[import-not-found]
    from ...types import PathLike
    from ...onnx.policy import OnnxPolicy
    import pandas as pd
    import gymnasium as gym

import math
from pathlib import Path

from benchmarl.experiment import Experiment  # type: ignore[import-not-found]

from ...config import GroupConfig
from ...indices import Indices
from ...parallel_env import MultiAgentNavgroundEnv
from .evaluate import evaluate_policy
from .navground_task import NavgroundTaskClass
from .policy import SingleAgentPolicy


class NavgroundExperiment(Experiment):
    """
    A :py:class:`benchmarl.experiment.Experiment` created from
    a :py:class:`navground.learning.parallel_env.MultiAgentNavgroundEnv`.

    :param env:                 The training environment

    :param task:                The task

    :param algorithm_config:    The algorithm configuration

    :param model_config:        The model configuration

    :param seed:                The seed

    :param config:              The experiment configuration

    :param critic_model_config: The critic model configuration

    :param callbacks:           The callbacks

    :param eval_env:            The evaluation environment. If
       not set, it will use the training environment for evaluation too.

    """

    def __init__(
        self,
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        seed: int,
        config: ExperimentConfig,
        env: MultiAgentNavgroundEnv | None = None,
        eval_env: MultiAgentNavgroundEnv | None = None,
        task: NavgroundTaskClass | None = None,
        critic_model_config: ModelConfig | None = None,
        callbacks: list[Callback] | None = None,
    ):
        if task is None and env is None:
            raise ValueError("At least one of env and task should be defined")
        if not task and env:
            task = NavgroundTaskClass('navground', {},
                                      env=env,
                                      eval_env=eval_env)
        super().__init__(task=task,
                         algorithm_config=algorithm_config,
                         model_config=model_config,
                         seed=seed,
                         config=config,
                         critic_model_config=critic_model_config,
                         callbacks=callbacks)

    def export_policy(self,
                      path: PathLike | None = None,
                      name: str = 'policy',
                      group: str = 'agent',
                      dynamic_batch_size: bool = True) -> Path:
        """
        Export the policy to onnx using :py:func:`navground.learning.onnx.export`.

        :param      path:                Where to save the policy. If not set,
                                         it save it in the experiment log folder
                                         under <name>_<group>.onnx.
        :param      name:                The name file (used as alternative to ``path``)
        :param      group:               The group name
        :param      dynamic_batch_size:  Whether to expose a dynamic batch size

        :returns:   The path of the saved file.
        """
        from ...onnx.export import export

        if path is None:
            path = Path(self.folder_name) / f"{name}_{group}.onnx"
        export(self.get_single_agent_policy(group),
               path=path,
               dynamic_batch_size=dynamic_batch_size)
        return Path(path)

    def export_policies(self,
                        path: PathLike | None = None,
                        name: str = 'policy',
                        dynamic_batch_size: bool = True) -> list[Path]:
        """
        Export the policy to onnx using :py:func:`navground.learning.onnx.export`.

        :param      path:                The directoru where to save the policies. If not set,
                                         it uses the experiment log folder.
        :param      name:                The files prefix
        :param      dynamic_batch_size:  Whether to expose a dynamic batch size

        :returns:   The path of the saved files.
        """
        from ...onnx.export import export

        if path is None:
            path = Path(self.folder_name)
        else:
            path = Path(path)
        paths = []
        for group in self.group_map:
            group_path = path / f"{name}_{group}.onnx"
            export(self.get_single_agent_policy(group),
                   path=group_path,
                   dynamic_batch_size=dynamic_batch_size)
            paths.append(group_path)
        return paths

    def load_policy(self,
                    path: PathLike | None = None,
                    name: str = 'policy',
                    group: str = 'agent') -> OnnxPolicy:
        """
        Loads an onnx policy.

        :param      path:  The path.
            If not set, it load <name>_<group>.onnx from the experiment log folder.

        :param      name:  The name file (used as alternative to ``path``)
        :param      group: The name of the group using the policy.
                           If provided it is used to associate an action space
                           to the policy.

        :returns:   The loaded policy.
        """
        from ...onnx.policy import OnnxPolicy

        if path is None:
            path = Path(self.folder_name) / f"{name}_{group}.onnx"

        if group:
            action_space = self.action_space(group)
        else:
            action_space = None
        return OnnxPolicy(path=path, action_space=action_space)

    def get_indices(self, group: str) -> list[int]:
        """
        Gets the indices of agents with name <group>_<index>

        :param      group:  The group

        :returns:   The agent indices
        """
        env = self.test_env._env
        return env.get_indices(group)

    def load_policies(self,
                      path: PathLike | None = None,
                      name: str = 'policy') -> list[GroupConfig]:
        """
        Loads the onnx policy for each group.

        :param      path:  The directory path.
            If not set, it uses the experiment log folder.

        :param      name:  The file prefix

        :returns:   A list of configurations with loaded policies
                    and agent indices set
        """
        return [
            GroupConfig(indices=Indices(self.get_indices(group)),
                        policy=self.load_policy(name="best_policy",
                                                group=group))
            for group in self.group_map
        ]

    def action_space(self, group: str) -> gym.Space:
        """
        Gets the action space associated with a group

        :param       group:  The group name

        :raises ValueError:  If the group is empty

        :returns:            The action space.
        """
        agents = self.group_map.get(group, [])
        if not agents:
            raise ValueError(f"Unknown group {group}!")
        env = self.test_env._env
        return env.action_space(agents[0])

    def observation_space(self, group: str) -> gym.Space:
        """
        Gets the observation space associated with a group

        :param       group:  The group name

        :raises ValueError:  If the group is empty

        :returns:            The action space.
        """
        agents = self.group_map.get(group, [])
        if not agents:
            raise ValueError(f"Unknown group {group}!")
        env = self.test_env._env
        return env.observation_space(agents[0])

    def get_single_agent_policy(self,
                                group: str = 'agent') -> SingleAgentPolicy:
        """
        Gets the policy associated with a group, as single agent
        (i.e., non-tensordict) policy.

        :param       group:  The group name

        :raises ValueError:  If the group is empty

        :returns:            The policy.
        """
        agents = self.group_map.get(group, [])
        if not agents:
            raise ValueError(f"Unknown group {group}!")
        env = self.test_env._env
        action_space = env.action_space(agents[0])
        observation_space = env.observation_space(agents[0])
        return SingleAgentPolicy(action_space=action_space,
                                 observation_space=observation_space,
                                 policy=self.group_policies[group])

    def get_single_agent_policies(self) -> list[GroupConfig]:
        """
        Gets the policy associated for each group, single agent
        (i.e., non-tensordict) policies.

        :returns:   A list of configurations with loaded policies
                    and agent indices set
        """
        env = self.test_env._env
        configs = []
        for group, agents in self.group_map.items():
            agent = agents[0]
            action_space = env.action_space(agent)
            observation_space = env.observation_space(agent)
            policy = SingleAgentPolicy(action_space=action_space,
                                       observation_space=observation_space,
                                       policy=self.group_policies[group])
            configs.append(
                GroupConfig(policy=policy,
                            indices=Indices(self.get_indices(group))))
        return configs

    def run_for(self, iterations: int = 0, steps: int = 0):
        """
        Train the policy for some iterations and/or steps.

        :param      iterations:  The iterations
        :param      steps:       The steps
        """
        n = self.config.collected_frames_per_batch(self.on_policy)
        target_iterations = self.n_iters_performed + max(
            iterations, int(math.ceil(steps / n)))
        target_steps = self.total_frames + max(steps, iterations * n)
        self.config.max_n_iters = max(target_iterations,
                                      self.config.max_n_iters)
        self.config.max_n_frames = max(target_steps, self.config.max_n_frames)
        self._setup_collector()
        self.run()

    @property
    def log_directory(self) -> Path:
        """
        The logging directory.

        :returns: The directory path
        """
        path = Path(self.folder_name)
        name = path.parts[-1]
        return path / name

    def load_log(self) -> pd.DataFrame:
        """
        Loads data stored in the the logs csv files.

        :returns:   A dataframe for all data.
        """
        import pandas as pd

        rs = []
        for path in (self.log_directory / 'scalars').glob('*.csv'):
            ds = pd.read_csv(path, index_col=[0], names=[str(path.stem)])
            rs.append(ds)
        return pd.concat(rs, axis=1)

    def evaluate_policy(
        self,
        n_eval_episodes: int = 10,
        return_episode_rewards: bool = False
    ) -> tuple[float, float] | tuple[list[float], list[int]]:
        """
        Evaluates the current policy using
        :py:func:navground.learning.utils.benchmarl.evaluate_policy`.

        :param      n_eval_episodes:         The number of episodes
        :param      return_episode_rewards:  Whether to return individual episode rewards
                                             (vs aggregate them)

        :returns:   If ``return_episode_rewards`` is set,
                    a tuple (list of cumulated episodes rewards', list of episodes length)
                    else, a tuple (average episodes rewards, std dev of episodes rewards)

        """
        return evaluate_policy(self.policy,
                               self.test_env,
                               n_eval_episodes=n_eval_episodes,
                               return_episode_rewards=return_episode_rewards)

    def plot_eval_logs(self,
                       group: str = 'agent',
                       two_axis: bool = False,
                       reward: bool = True,
                       reward_color: str = 'k',
                       reward_linestyle: str = '--',
                       success: bool = False,
                       length: bool = False,
                       reward_low: float | None = None,
                       reward_high: float = 0,
                       length_low: float = 0,
                       length_high: float | None = None,
                       **kwargs: Any) -> None:
        """
        Plots reward, success and/or length from the eval logs

        :param      group:    The name of the group
        :param      reward:       Whether to plot the reward
        :param      reward_color: The reward color
        :param      reward_linestyle: The reward linestyle
        :param      success:      Whether to plot the success
        :param      length:       Whether to plot the length
        :param      reward_low:   An optional lower bound to scale the reward
        :param      reward_high:  An optional upper bound to scale the reward
        :param      lenght_low:   An optional lower bound to scale the reward
        :param      lenght_high:  An optional upper bound to scale the reward
        :param      two_axis:     Whether to use two axis (only if there are two fields)
        """
        from ..plot import LogField, plot_logs

        fields: list[LogField] = []
        if reward:
            fields.append(
                LogField(label="mean reward",
                         key=f"eval_{group}_reward_episode_reward_mean",
                         linestyle=reward_linestyle,
                         color=reward_color,
                         low=reward_low,
                         high=reward_high))
        if success:
            fields.append(
                LogField(label="success rate",
                         key=f"eval_{group}_success_rate",
                         linestyle='-',
                         color='g'))
        if length:
            fields.append(
                LogField(label="mean length",
                         key="eval_reward_episode_len_mean",
                         linestyle=':',
                         color='b',
                         low=length_low,
                         high=length_high))
        plot_logs(self.load_log(),
                  two_axis=two_axis,
                  fields=fields,
                  key='counters_total_frames',
                  **kwargs)

    # Same as Experiment.reload_from_file but loading a NavgroundExperiment.
    @staticmethod
    def reload_from_file(restore_file: str) -> NavgroundExperiment:
        """
        Restores the experiment from the checkpoint file.

        This method expects the same folder structure created when an experiment is run.
        The checkpoint file (``restore_file``) is in the checkpoints directory
        and a config.pkl file is
        present a level above at ``restore_file/../../config.pkl``

        :param restore_file: The checkpoint file (.pt) of the experiment reload.

        :returns: The reloaded experiment.

        """
        import os
        import pickle

        experiment_folder = Path(restore_file).parent.parent.resolve()
        config_file = experiment_folder / "config.pkl"
        if not os.path.exists(config_file):
            raise ValueError("config.pkl file not found in experiment folder.")
        with open(config_file, "rb") as f:
            task = pickle.load(f)
            task_config = pickle.load(f)
            algorithm_config = pickle.load(f)
            model_config = pickle.load(f)
            seed = pickle.load(f)
            experiment_config = pickle.load(f)
            critic_model_config = pickle.load(f)
            callbacks = pickle.load(f)
        task.config = task_config
        experiment_config.restore_file = restore_file
        experiment = NavgroundExperiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=critic_model_config,
        )
        # print(f"\nReloaded experiment {experiment.name} from {restore_file}.")
        return experiment
