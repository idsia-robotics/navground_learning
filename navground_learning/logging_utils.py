from dataclasses import dataclass
from itertools import islice
from types import MethodType
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
import yaml
from imitation.algorithms.bc import BCLogger
from navground import sim
from navground.sim.pyplot_helpers import plot_runs
from navground.sim.ui.video import make_video_from_run
# from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.logger import (Figure, HParam, Logger,
                                             TensorBoardOutputFormat, Video)
from torch.utils.tensorboard.writer import SummaryWriter

from .env.base import NavgroundBaseEnv
from .evaluate import make_experiment_with_env
from .il.base_trainer import BaseTrainer


def log_yaml(writer: SummaryWriter, name: str, data: dict[str, Any]) -> None:
    text = f"```yaml\n{yaml.safe_dump(data)}```"
    writer.add_text(name, text, 0)
    writer.flush()


def log_env(writer: SummaryWriter, env: NavgroundBaseEnv) -> None:
    log_yaml(writer, "environment", env.asdict)


def log_graph(writer: SummaryWriter, env: NavgroundBaseEnv, policy: Any, index: int = 0) -> None:
    x = env._observation_space[index].sample()
    x = np.expand_dims(x, axis=0)
    x = th.from_numpy(x)
    writer.add_graph(policy.actor, x, use_strict_trace=True, verbose=False)


def get_tb_logger(logger: Logger) -> TensorBoardOutputFormat | None:
    fs = [
        f for f in logger.output_formats
        if isinstance(f, TensorBoardOutputFormat)
    ]
    if fs:
        return fs[0]
    return None


@dataclass
class VideoConfig:
    number: int = 0
    fps: int = 30
    factor: float = 1.0


@dataclass
class TrajectoryPlotConfig:
    number: int = 0
    columns: int = 1
    color: Callable[[sim.Agent], str] | None = None
    step: int = 10
    width: float = 10.0


def record_values(logger: Logger, key: str, values: np.ndarray) -> None:
    logger.record(f"eval/{key}/min", np.min(values))
    logger.record(f"eval/{key}/max", np.max(values))
    logger.record(f"eval/{key}/mean", np.mean(values))
    logger.record(f"eval/{key}/std_dev", np.std(values))


class LogTrajectories:

    def __init__(self,
                 video_config=VideoConfig(),
                 plot_config=TrajectoryPlotConfig(),
                 episodes: int = 100,
                 hparams: dict[str, Any] = {},
                 data: dict[str, Any] = {},
                 log_graph: bool = False):
        self.plot_config = plot_config
        self.video_config = video_config
        self.episodes = episodes
        self.hparams = hparams
        self.data = data
        self.log_graph = log_graph

    def init(self, env: NavgroundBaseEnv, policy: Any, logger: Logger) -> None:
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
            metric_dict = {"eval/reward/mean": 0.0, "eval/reward/min": 0.0}
            self.logger.record("hparams",
                               HParam(self.hparams, metric_dict),
                               exclude=("stdout", "log", "json", "csv"))

    def _init_logger(self, logger: Logger) -> None:
        self.logger = logger
        formats = [
            fmt for fmt in logger.output_formats
            if isinstance(fmt, TensorBoardOutputFormat)
        ]
        self.tb_writer = formats[0].writer if formats else None

    def _init_eval_exp(self, env, policy):
        self.eval_exp = make_experiment_with_env(env=env, policy=policy)
        self.eval_exp.number_of_runs = max(0, self.plot_config.number,
                                           self.video_config.number,
                                           self.episodes)
        self.eval_exp.record_config.pose = True
        self.eval_exp.record_config.collisions = True
        self.eval_exp.record_config.efficacy = True
        self.eval_exp.record_config.safety_violation = True
        # self.eval_exp.record_config.deadlocks = True

    def evaluate(self, global_step: int) -> None:
        self.eval_exp.remove_all_runs()
        self.eval_exp.run()
        if self.plot_config.number > 0:
            runs = list(self.eval_exp.runs.values())[:self.plot_config.number]
            fig = plot_runs(runs,
                            columns=self.plot_config.columns,
                            step=self.plot_config.step,
                            agent_color=lambda a: self.plot_config.color,
                            width=self.plot_config.width)
            self.logger.record("eval/figure",
                               Figure(fig, close=True),
                               exclude=("stdout", "log", "json", "csv"))
        if self.video_config.number > 0:
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
            self.tb_writer.add_histogram('eval/reward/distribution',
                                         values=rewards,
                                         global_step=global_step)
        sv = np.array([
            np.sum(run.safety_violations)
            for run in self.eval_exp.runs.values()
        ])
        record_values(self.logger, "safety_violations", sv)
        efficacy = np.asarray(
            [np.mean(run.efficacy) for run in self.eval_exp.runs.values()])
        record_values(self.logger, "efficacy", efficacy)
        collisions = np.asarray(
            [len(run.collisions) for run in self.eval_exp.runs.values()])
        record_values(self.logger, "collisions", collisions)
        # deadlocks = np.asarray(
        #     [sum(run.deadlocks >= 0) for run in self.eval_exp.runs.values()])
        # record_values(self.logger, "deadlocks", deadlocks)


def config_log(model: OffPolicyAlgorithm,
               env: Any = None,
               **kwargs: Any) -> None:
    # not working for pz wrapped envs
    if env is None:
        venv = model.get_env()
        # TODO(Jerome): handle venvs
        try:
            env = venv.envs[0]  # type: ignore
        except:
            pass
    if env:
        env = env.unwrapped
    if not isinstance(env, NavgroundBaseEnv):
        raise TypeError("Not a navground env")
    log = LogTrajectories(**kwargs)
    log.init(policy=model.policy, logger=model.logger, env=env)
    _dump_logs = model._dump_logs

    def dump_logs(model: OffPolicyAlgorithm) -> None:
        log.evaluate(global_step=model.num_timesteps)
        _dump_logs()

    model._dump_logs = MethodType(dump_logs, model)  # type: ignore


def config_il_log(trainer: BaseTrainer, **kwargs: Any) -> None:
    log = LogTrajectories(**kwargs)
    log.init(policy=trainer.policy,
             logger=trainer.logger.default_logger,
             env=trainer.env)
    bc_logger = trainer.bc_trainer._bc_logger
    _log_batch = bc_logger.log_batch

    def log_batch(logger: BCLogger, batch_num: int, batch_size: int,
                  num_samples_so_far: int, *args: Any, **kwargs: Any) -> None:
        log.evaluate(global_step=logger._tensorboard_step)
        _log_batch(batch_num, batch_size, num_samples_so_far, *args, **kwargs)

    bc_logger.log_batch = MethodType(log_batch, bc_logger)  # type: ignore
