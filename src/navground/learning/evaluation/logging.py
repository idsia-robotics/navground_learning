from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import islice
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from pettingzoo.utils.env import ParallelEnv

from navground import sim

from ..config import GroupConfig
from ..env import BaseEnv
from ..internal.base_env import NavgroundBaseEnv
from ..types import (AnyPolicyPredictor, GroupObservationsTransform,
                     ObservationTransform, PathLike)
from .experiment import make_experiment_with_env

if TYPE_CHECKING:
    from imitation.algorithms.bc import BCLogger
    from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
    from stable_baselines3.common.off_policy_algorithm import \
        OffPolicyAlgorithm
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv
    from torch.utils.tensorboard.writer import SummaryWriter

    from ..il.base import BaseILAlgorithm
    from ..parallel_env import BaseParallelEnv


def log_yaml(writer: SummaryWriter, name: str, data: dict[str, Any]) -> None:
    text = f"```yaml\n{yaml.safe_dump(data)}```"
    writer.add_text(name, text, 0)  # type: ignore[no-untyped-call]
    writer.flush()  # type: ignore[no-untyped-call]


def log_env(writer: SummaryWriter,
            env: BaseEnv | VecEnv | BaseParallelEnv) -> None:
    from stable_baselines3.common.vec_env import VecEnv

    if isinstance(env, VecEnv):
        value = env.get_attr("asdict", [0])[0]
    elif isinstance(env, ParallelEnv):
        if isinstance(env, NavgroundBaseEnv):
            value = env.asdict
        elif isinstance(env.unwrapped, NavgroundBaseEnv):
            value = env.unwrapped.asdict
        else:
            raise TypeError(f"Env type {type(env)} not supported")
    else:
        value = env.get_wrapper_attr("asdict")
    log_yaml(writer, "environment", value)


def log_graph(writer: SummaryWriter,
              env: BaseEnv | VecEnv | BaseParallelEnv,
              policy: Any,
              index: int = 0) -> None:
    import torch as th

    if isinstance(env, ParallelEnv):
        obs = env.observation_space(0).sample()
    else:
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
    tensorboard: bool = True
    """whether to add to the tensorboard log"""
    file: bool = True
    """whether to add to save to a file"""
    format: str = "mp4"
    """video file format"""


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
    world_kwargs: dict[str, Any] = field(default_factory=dict)
    """The world kwargs passed to :py:func:navground.sim.pyplot_helpers.plot_runs"""
    hide_axes: bool = True
    """whether to hide the axis"""
    tensorboard: bool = True
    """whether to add to the tensorboard log"""
    file: bool = True
    """whether to add to save to a file"""


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
                 safety_violation: bool = True,
                 duration: bool = False,
                 processes: int = 1,
                 use_multiprocess: bool = False,
                 use_onnx: bool = True,
                 grouped: bool = False,
                 pre: ObservationTransform | None = None,
                 group_pre: GroupObservationsTransform | None = None,
                 every: int = 1):
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
        self.record_duration = duration
        self.processes = processes
        self.use_multiprocess = use_multiprocess
        self.use_onnx = use_onnx or self.processes > 1
        self.grouped = grouped
        self.pre = pre
        self.group_pre = group_pre
        self.every = every
        self._count = 0

    @property
    def record_pose(self) -> bool:
        return self.plot_config.number > 0 or self.video_config.number > 0

    @property
    def number_of_runs(self) -> int:
        return max(0, self.plot_config.number, self.video_config.number,
                   self.episodes)

    def init(self, env: BaseEnv | VecEnv | BaseParallelEnv, policy: BasePolicy,
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

    def _init_eval_exp(self, env: BaseEnv | VecEnv | BaseParallelEnv,
                       policy: BasePolicy) -> None:
        if self.use_onnx:
            _policy: AnyPolicyPredictor | PathLike = "_policy.onnx"
            self._policy = policy
        else:
            _policy = policy

        group = GroupConfig(policy=_policy, color=self.video_config.color)
        exp = make_experiment_with_env(env=env,
                                       groups=[group],
                                       record_reward=self.record_reward,
                                       grouped=self.grouped,
                                       pre=self.pre,
                                       group_pre=self.group_pre)
        self.eval_exp = exp
        self.eval_exp.number_of_runs = self.number_of_runs
        self.eval_exp.record_config.pose = self.record_pose
        self.eval_exp.record_config.collisions = self.record_collisions
        self.eval_exp.record_config.efficacy = self.record_efficacy
        self.eval_exp.record_config.safety_violation = self.record_safety_violation
        if self.plot_config.number > 0 or self.video_config.number > 0:
            self.eval_exp.record_config.world = True
        # self.eval_exp.record_config.deadlocks = True

    @property
    def video_directory(self) -> pathlib.Path:
        d = self.logger.get_dir()
        if d is None:
            raise ValueError("Logging directory not set")
        path = pathlib.Path(d) / "videos"
        path.mkdir(exist_ok=True)
        return path

    @property
    def plot_directory(self) -> pathlib.Path:
        d = self.logger.get_dir()
        if d is None:
            raise ValueError("Logging directory not set")
        path = pathlib.Path(d) / "figures"
        path.mkdir(exist_ok=True)
        return path

    def evaluate(self, global_step: int) -> None:
        self._count += 1
        if self._count % self.every:
            return
        if self.eval_exp.number_of_runs:
            if self.use_onnx:
                from ..onnx import export
                export(self._policy, "_policy.onnx")

            self.eval_exp.remove_all_runs()
            if self.processes > 1:
                self.eval_exp.run_mp(number_of_processes=self.processes,
                                     keep=True,
                                     use_multiprocess=self.use_multiprocess)
            else:
                self.eval_exp.run()
        else:
            return
        if self.plot_config.number > 0:
            from matplotlib import pyplot as plt
            from stable_baselines3.common.logger import Figure

            from navground.sim.pyplot_helpers import plot_runs

            runs = list(self.eval_exp.runs.values())[:self.plot_config.number]
            fig = plot_runs(runs,
                            columns=self.plot_config.columns,
                            step=self.plot_config.step,
                            with_agent=True,
                            with_world=True,
                            color=self.plot_config.color,
                            width=self.plot_config.width,
                            world_kwargs=self.plot_config.world_kwargs,
                            hide_axes=self.plot_config.hide_axes)
            if self.plot_config.tensorboard:
                self.logger.record("eval/figure",
                                   Figure(fig, close=True),
                                   exclude=("stdout", "log", "json", "csv"))
            if self.plot_config.file:
                path = self.plot_directory / f"figure_{global_step}.pdf"
                plt.savefig(str(path))
            plt.close()

        if self.video_config.number > 0:
            import torch as th
            from stable_baselines3.common.logger import Video

            from navground.sim.ui.video import (MOVIEPY_VERSION,
                                                make_video_from_run)

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
                if self.video_config.tensorboard:
                    self.logger.record(f"eval/video_{i}",
                                       Video(th.from_numpy(frames),
                                             fps=self.video_config.fps),
                                       exclude=("stdout", "log", "json",
                                                "csv"))
                if self.video_config.file:
                    path = (
                        self.video_directory /
                        f"video_{global_step}_{i}.{self.video_config.format}")
                    if path.suffix.lower() == ".gif":
                        video.write_gif(str(path), fps=self.video_config.fps)
                    else:
                        kwargs = {'audio': False, 'logger': None}
                        if MOVIEPY_VERSION == 1:
                            kwargs['verbose'] = False
                        video.write_videofile(str(path),
                                              fps=self.video_config.fps,
                                              **kwargs)

        steps = [run.recorded_steps for run in self.eval_exp.runs.values()]
        record_values(self.logger, "steps", np.asarray(steps))

        success = [
            np.mean(np.asarray(run.get_record("success")) > 0)
            for run in self.eval_exp.runs.values() if "success" in run.records
        ]
        if success:
            record_values(self.logger, "success", np.asarray(success))

        rewards = np.array([
            np.sum(run.get_record("reward"), axis=0)
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
                np.sum(run.safety_violations, axis=0)
                for run in self.eval_exp.runs.values()
            ])
            record_values(self.logger, "safety_violations", sv)
        if self.record_efficacy:
            efficacy = np.asarray([
                np.mean(run.efficacy, axis=0)
                for run in self.eval_exp.runs.values()
            ])
            record_values(self.logger, "efficacy", efficacy)
        if self.record_collisions:
            collisions = np.asarray(
                [len(run.collisions) for run in self.eval_exp.runs.values()])
            record_values(self.logger, "collisions", collisions)
        if self.record_duration:
            duration = np.asarray(
                [run.final_sim_time for run in self.eval_exp.runs.values()])
            record_values(self.logger, "duration", duration)
        # deadlocks = np.asarray(
        #     [sum(run.deadlocks >= 0) for run in self.eval_exp.runs.values()])
        # record_values(self.logger, "deadlocks", deadlocks)


def config_eval_log(
        model: OffPolicyAlgorithm | BaseILAlgorithm,
        env: BaseEnv | VecEnv | BaseParallelEnv | None = None,
        video_config: VideoConfig = VideoConfig(),
        plot_config: TrajectoryPlotConfig = TrajectoryPlotConfig(),
        every: int = 1,
        episodes: int = 100,
        hparams: dict[str, Any] = {},
        data: dict[str, Any] = {},
        log_graph: bool = False,
        reward: bool = True,
        collisions: bool = True,
        efficacy: bool = True,
        safety_violation: bool = True,
        duration: bool = False,
        processes: int = 1,
        use_multiprocess: bool = False,
        use_onnx: bool = True,
        grouped: bool = False,
        pre: ObservationTransform | None = None,
        group_pre: GroupObservationsTransform | None = None) -> None:
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
    :param      collisions:        Whether to record episodes' collisions
    :param      efficacy:          Whether to record episodes' efficacy
    :param      safety_violation:  Whether to record episodes' safety violation
    :param      duration:          Whether to record episodes' duration
    :param      processes:         Number of processes to use
    :param      use_multiprocess:  Whether to use ``multiprocess`` instead of ``multiprocessing``
    :param      use_onnx:          Whether to use onnx for inference
    :param      grouped:           Whether the policy is grouped.
    :param      pre:               An optional transformation to apply to observations
                                   of all individual agents
    :param      group_pre:         An optional transformation to apply to observations
                                   of all groups
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
                  safety_violation=safety_violation,
                  duration=duration,
                  processes=processes,
                  use_multiprocess=use_multiprocess,
                  use_onnx=use_onnx,
                  grouped=grouped,
                  pre=pre,
                  group_pre=group_pre,
                  every=every)
    if isinstance(model, OffPolicyAlgorithm):
        logger = model.logger
    else:
        logger = model.logger.default_logger
    log.init(policy=model.policy, logger=logger, env=env)
    if isinstance(model, OffPolicyAlgorithm):
        _config_sb3(model, log)
    else:
        _config_il(model, log)


# In SB2.6 _dump_logs has been renamed to dump_logs
def _config_sb3(model: OffPolicyAlgorithm, log: EvalLog) -> None:
    _dump_logs = model.dump_logs
    _excluded_save_params = model._excluded_save_params

    def dump_logs(model: OffPolicyAlgorithm) -> None:
        log.evaluate(global_step=model.num_timesteps)
        _dump_logs()

    def excluded_save_params(self: OffPolicyAlgorithm) -> list[str]:
        return _excluded_save_params() + ['dump_logs', '_excluded_save_params']

    model.dump_logs = MethodType(dump_logs, model)  # type: ignore
    model._excluded_save_params = MethodType(  # type: ignore[method-assign]
        excluded_save_params, model)


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
