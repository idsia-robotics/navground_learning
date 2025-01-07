from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from navground import sim

from ..config import GroupConfig
from ..env import BaseEnv
from ..types import AnyPolicyPredictor
from .experiment import make_experiment_with_env

if TYPE_CHECKING:
    from imitation.algorithms.bc import BCLogger
    from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
    from stable_baselines3.common.off_policy_algorithm import \
        OffPolicyAlgorithm
    from stable_baselines3.common.vec_env import VecEnv
    from torch.utils.tensorboard.writer import SummaryWriter

    from ..il.base import BaseILAlgorithm


def log_yaml(writer: SummaryWriter, name: str, data: dict[str, Any]) -> None:
    text = f"```yaml\n{yaml.safe_dump(data)}```"
    writer.add_text(name, text, 0)  # type: ignore[no-untyped-call]
    writer.flush()  # type: ignore[no-untyped-call]


def log_env(writer: SummaryWriter, env: BaseEnv | VecEnv) -> None:
    from stable_baselines3.common.vec_env import VecEnv

    if isinstance(env, VecEnv):
        value = env.get_attr("asdict", [0])[0]
    else:
        value = env.get_wrapper_attr("asdict")
    log_yaml(writer, "environment", value)


def log_graph(writer: SummaryWriter,
              env: BaseEnv | VecEnv,
              policy: Any,
              index: int = 0) -> None:
    import torch as th

    obs = env.observation_space.sample()
    if isinstance(obs, dict):
        x: th.Tensor | dict[str, th.Tensor] = {
            k: th.from_numpy(v[np.newaxis, :])
            for k, v in obs.values()
        }
    else:
        x = th.from_numpy(obs[np.newaxis, :])
    writer.add_graph(policy.actor, x, use_strict_trace=True,
                     verbose=False)  # type: ignore[no-untyped-call]


def get_tb_logger(logger: Logger) -> TensorBoardOutputFormat | None:
    from stable_baselines3.common.logger import TensorBoardOutputFormat
    fs = [
        f for f in logger.output_formats
        if isinstance(f, TensorBoardOutputFormat)
    ]
    if fs:
        return fs[0]
    return None


@dataclass
class VideoConfig:
    """
    The configuration to log videos
    """
    number: int = 0
    """number of runs per logging update to make a video from"""
    fps: int = 30
    """fps"""
    factor: float = 1.0
    """real-time factor for the videos"""
    color: str = ''
    """color of the agents that are learning"""


@dataclass
class TrajectoryPlotConfig:
    """
    The configuration to log trajectory plots
    """
    number: int = 0
    """number of runs whose trajectory to plot"""
    columns: int = 1
    """number of columns"""
    color: Callable[[sim.Agent], str] | None = None
    """color of the agents"""
    step: int = 10
    """time steps interval at which to draw the agent in the plots"""
    width: float = 10.0
    """total width of the plots"""


def record_values(logger: Logger, key: str,
                  values: np.typing.NDArray[np.floating[Any]]) -> None:
    logger.record(f"eval/{key}/min", np.min(values))
    logger.record(f"eval/{key}/max", np.max(values))
    logger.record(f"eval/{key}/mean", np.mean(values))
    logger.record(f"eval/{key}/std_dev", np.std(values))


class EvalLog:

    def __init__(self,
                 video_config: VideoConfig = VideoConfig(),
                 plot_config: TrajectoryPlotConfig = TrajectoryPlotConfig(),
                 episodes: int = 100,
                 hparams: dict[str, Any] = {},
                 data: dict[str, Any] = {},
                 log_graph: bool = False,
                 reward: bool = True,
                 collisions: bool = True,
                 efficacy: bool = True,
                 safety_violation: bool = True):
        self.plot_config = plot_config
        self.video_config = video_config
        self.episodes = episodes
        self.hparams = hparams
        self.data = data
        self.log_graph = log_graph
        self.record_collisions = collisions
        self.record_safety_violation = safety_violation
        self.record_efficacy = efficacy
        self.record_reward = reward

    @property
    def record_pose(self) -> bool:
        return self.plot_config.number > 0 or self.video_config.number > 0

    @property
    def number_of_runs(self) -> int:
        return max(0, self.plot_config.number, self.video_config.number,
                   self.episodes)

    def init(self, env: BaseEnv | VecEnv, policy: AnyPolicyPredictor,
             logger: Logger) -> None:
        self._init_logger(logger)
        self._init_eval_exp(env=env, policy=policy)
        if self.tb_writer:
            if self.log_graph:
                log_graph(self.tb_writer, env, policy)
            log_env(self.tb_writer, env)
            if self.data:
                for key, value in self.data.items():
                    log_yaml(self.tb_writer, key, value)
        if self.hparams:
            from stable_baselines3.common.logger import HParam

            metric_dict = {"eval/reward/mean": 0.0, "eval/reward/min": 0.0}
            self.logger.record("hparams",
                               HParam(self.hparams, metric_dict),
                               exclude=("stdout", "log", "json", "csv"))

    def _init_logger(self, logger: Logger) -> None:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        self.logger = logger
        formats = [
            fmt for fmt in logger.output_formats
            if isinstance(fmt, TensorBoardOutputFormat)
        ]
        self.tb_writer = formats[0].writer if formats else None

    def _init_eval_exp(self, env: BaseEnv | VecEnv, policy: AnyPolicyPredictor) -> None:
        group = GroupConfig(policy=policy, color=self.video_config.color)
        exp = make_experiment_with_env(env=env,
                                       groups=[group],
                                       record_reward=self.record_reward)
        self.eval_exp = exp
        self.eval_exp.number_of_runs = self.number_of_runs
        self.eval_exp.record_config.pose = self.record_pose
        self.eval_exp.record_config.collisions = self.record_collisions
        self.eval_exp.record_config.efficacy = self.record_efficacy
        self.eval_exp.record_config.safety_violation = self.record_safety_violation
        # self.eval_exp.record_config.deadlocks = True

    def evaluate(self, global_step: int) -> None:
        if self.eval_exp.number_of_runs:
            self.eval_exp.remove_all_runs()
            self.eval_exp.run()
        else:
            return
        if self.plot_config.number > 0:
            from stable_baselines3.common.logger import Figure
            from navground.sim.pyplot_helpers import plot_runs

            runs = list(self.eval_exp.runs.values())[:self.plot_config.number]
            fig = plot_runs(runs,
                            columns=self.plot_config.columns,
                            step=self.plot_config.step,
                            with_agent=True,
                            with_world=True,
                            color=lambda a: self.plot_config.color,
                            width=self.plot_config.width)
            self.logger.record("eval/figure",
                               Figure(fig, close=True),
                               exclude=("stdout", "log", "json", "csv"))
        if self.video_config.number > 0:
            import torch as th
            from stable_baselines3.common.logger import Video
            from navground.sim.ui.video import make_video_from_run

            for i, run in islice(self.eval_exp.runs.items(),
                                 self.video_config.number):

                video = make_video_from_run(run,
                                            factor=self.video_config.factor)
                frames = np.array(
                    list(video.iter_frames(fps=self.video_config.fps)))
                # to (T, C, H, W)
                frames = frames.transpose(0, 3, 1, 2)
                # to (1=B, T, C, H, W)
                frames = np.expand_dims(frames, axis=0)
                self.logger.record(f"eval/video_{i}",
                                   Video(th.from_numpy(frames),
                                         fps=self.video_config.fps),
                                   exclude=("stdout", "log", "json", "csv"))
        rewards = np.array([
            np.sum(run.get_record("reward"))
            for run in self.eval_exp.runs.values()
        ])
        record_values(self.logger, "reward", rewards)
        if self.tb_writer:
            self.tb_writer.add_histogram(
                'eval/reward/distribution',  # type: ignore[no-untyped-call]
                values=rewards,
                global_step=global_step)
        if self.record_safety_violation:
            sv = np.array([
                np.sum(run.safety_violations)
                for run in self.eval_exp.runs.values()
            ])
            record_values(self.logger, "safety_violations", sv)
        if self.record_efficacy:
            efficacy = np.asarray(
                [np.mean(run.efficacy) for run in self.eval_exp.runs.values()])
            record_values(self.logger, "efficacy", efficacy)
        if self.record_collisions:
            collisions = np.asarray(
                [len(run.collisions) for run in self.eval_exp.runs.values()])
            record_values(self.logger, "collisions", collisions)
        # deadlocks = np.asarray(
        #     [sum(run.deadlocks >= 0) for run in self.eval_exp.runs.values()])
        # record_values(self.logger, "deadlocks", deadlocks)


def config_eval_log(model: OffPolicyAlgorithm | BaseILAlgorithm,
                    env: BaseEnv | VecEnv | None = None,
                    video_config: VideoConfig = VideoConfig(),
                    plot_config: TrajectoryPlotConfig = TrajectoryPlotConfig(),
                    episodes: int = 100,
                    hparams: dict[str, Any] = {},
                    data: dict[str, Any] = {},
                    log_graph: bool = False,
                    reward: bool = True,
                    collisions: bool = True,
                    efficacy: bool = True,
                    safety_violation: bool = True) -> None:
    """
    Configure the model logger to log additional data:

    - trajectory plots
    - trajectory videos
    - statistics on reward, collisions, efficacy and safety violations
    - hparams
    - data as YAML
    - model policy graph
    - a YAML representation of the environment (at the begin of the logging)

    :param      model:             The model being trained
    :param      env:               The testing environment
    :param      video_config:      The video configuration
    :param      plot_config:       The plot configuration
    :param      episodes:          The number of episodes over which to compute statistics
    :param      hparams:           The hparams
    :param      data:              The data
    :param      log_graph:         The log graph
    :param      collisions:        The collisions
    :param      efficacy:          The efficacy
    :param      safety_violation:  The safety violation
    """
    from stable_baselines3.common.off_policy_algorithm import \
        OffPolicyAlgorithm

    if env is None:
        env = model.get_env()
    assert env is not None
    log = EvalLog(video_config=video_config,
                  plot_config=plot_config,
                  episodes=episodes,
                  hparams=hparams,
                  data=data,
                  log_graph=log_graph,
                  reward=reward,
                  collisions=collisions,
                  efficacy=efficacy,
                  safety_violation=safety_violation)
    if isinstance(model, OffPolicyAlgorithm):
        logger = model.logger
    else:
        logger = model.logger.default_logger
    log.init(policy=model.policy, logger=logger, env=env)
    if isinstance(model, OffPolicyAlgorithm):
        _config_sb3(model, log)
    else:
        _config_il(model, log)


def _config_sb3(model: OffPolicyAlgorithm, log: EvalLog) -> None:
    _dump_logs = model._dump_logs

    def dump_logs(model: OffPolicyAlgorithm) -> None:
        log.evaluate(global_step=model.num_timesteps)
        _dump_logs()

    model._dump_logs = MethodType(dump_logs, model)  # type: ignore


def _config_il(model: BaseILAlgorithm, log: EvalLog) -> None:
    if not model.logger:
        return

    from ..il import BC, DAgger

    if isinstance(model, BC):
        bc_logger = model._trainer._bc_logger
    elif isinstance(model, DAgger):
        bc_logger = model._bc_trainer._bc_logger
    else:
        return

    _log_batch = bc_logger.log_batch

    def log_batch(logger: BCLogger, batch_num: int, batch_size: int,
                  num_samples_so_far: int, *args: Any, **kwargs: Any) -> None:
        log.evaluate(global_step=logger._tensorboard_step)
        _log_batch(batch_num, batch_size, num_samples_so_far, *args, **kwargs)

    bc_logger.log_batch = MethodType(log_batch, bc_logger)
