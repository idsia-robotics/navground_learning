from __future__ import annotations

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

import numpy as np

from navground import sim
from navground.sim.ui.video import display_video, record_video

from ..config import GroupConfig
from .experiment import make_experiment_with_env

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

    from ..env import BaseEnv
    from ..parallel_env import BaseParallelEnv
    from ..types import (AnyPolicyPredictor, GroupObservationsTransform,
                         ObservationTransform, PathLike)


def _run_video(
        fn: Callable,  # type: ignore[type-arg]
        experiment: sim.Experiment,
        factor: int = 1,
        seed: int = 0,
        use_world_bounds: bool = True,
        select: int = 0,
        of: int = 0,
        **kwargs: Any) -> Any:
    if of:
        experiment.run_index = seed
        experiment.run(number_of_runs=of)
        rs = sorted(((np.min(np.sum(run.records['reward'], axis=0)), i)
                     for i, run in experiment.runs.items()),
                    reverse=True)
        index = rs[select][1]
        run = experiment.runs[index]
    else:
        run = experiment.run_once(seed)
    world = experiment.scenario.make_world(run.seed)
    if use_world_bounds:
        bounds = world.bounds
    else:
        bounds = None
    if run.final_sim_time:
        r = fn(world=world,
               time_step=run.time_step,
               duration=run.final_sim_time,
               factor=factor,
               fps=factor / run.time_step,
               bounds=bounds,
               **kwargs)
    else:
        r = None
    world._close()
    return r


def display_run_video(experiment: sim.Experiment,
                      factor: int = 1,
                      seed: int = 0,
                      use_world_bounds: bool = True,
                      select: int = 0,
                      of: int = 0,
                      **kwargs: Any) -> Any:
    """
    Display a video from one episode from an evaluation experiment.

    :param      experiment:        The experiment
    :param      factor:            The real-time factor
    :param      seed:              The seed (only applies if ``of == 0``)
    :param      use_world_bounds:  Whether to keep the initial world bounds
    :param      select:            Which episode to select
                                   (ordered by reward, only applies if ``of > 0``)
    :param      of:                The number of runs to choose the episode from
    :param      kwargs:            Keywords arguments passed to the renderer
    """
    return _run_video(display_video,
                      experiment,
                      factor=factor,
                      seed=seed,
                      use_world_bounds=use_world_bounds,
                      select=select,
                      of=of,
                      **kwargs)


def record_run_video(experiment: sim.Experiment,
                     path: PathLike,
                     factor: int = 1,
                     seed: int = 0,
                     use_world_bounds: bool = True,
                     select: int = 0,
                     of: int = 0,
                     **kwargs: Any) -> None:
    """
    Record a video from one episode from an evaluation experiment.

    :param      experiment:        The experiment
    :param      path:              Where to save the video
    :param      factor:            The real-time factor
    :param      seed:              The seed (only applies if ``of == 0``)
    :param      use_world_bounds:  Whether to keep the initial world bounds
    :param      select:            Which episode to select
                                   (ordered by reward, only applies if ``of > 0``)
    :param      of:                The number of runs to choose the episode from
    :param      kwargs:            Keywords arguments passed to the renderer

    """
    _run_video(record_video,
               experiment,
               path=path,
               factor=factor,
               seed=seed,
               use_world_bounds=use_world_bounds,
               select=select,
               of=of,
               **kwargs)


def display_episode_video(env: BaseEnv | BaseParallelEnv | VecEnv,
                          groups: Collection[GroupConfig] = tuple(),
                          policy: AnyPolicyPredictor | PathLike = '',
                          factor: int = 1,
                          seed: int = 0,
                          grouped: bool = False,
                          pre: ObservationTransform | None = None,
                          group_pre: GroupObservationsTransform | None = None,
                          use_world_bounds: bool = True,
                          of: int = 1,
                          select: int = 0,
                          **kwargs: Any) -> Any:
    """
    Display the video from one episode of an environment.

    :param      env:               The environment
    :param      groups:            The configuration of the groups
    :param      policy:            The default policy
                                   (when not specified in the group config)
    :param      factor:            The real-time factor
    :param      seed:              The seed (only applies if ``of == 0``)
    :param      grouped:           Whether the policy is grouped
    :param      pre:               An optional transformation to apply to observations
                                   of all individual agents
    :param      group_pre:         An optional transformation to apply to observations
                                   of all groups
    :param      use_world_bounds:  Whether to keep the initial world bounds
    :param      select:            Which episode to select
                                   (ordered by reward, only applies if ``of > 0``)
    :param      of:                The number of runs to choose the episode from
    :param      kwargs:            Keywords arguments passed to the renderer

    """
    exp = make_experiment_with_env(env,
                                   groups=groups,
                                   policy=policy,
                                   record_reward=of > 0,
                                   grouped=grouped,
                                   pre=pre,
                                   group_pre=group_pre,
                                   record_success=False)
    return display_run_video(exp,
                             factor=factor,
                             seed=seed,
                             select=select,
                             of=of,
                             **kwargs)


def record_episode_video(env: BaseEnv | BaseParallelEnv | VecEnv,
                         path: PathLike,
                         groups: Collection[GroupConfig] = tuple(),
                         policy: AnyPolicyPredictor | PathLike = '',
                         factor: int = 1,
                         seed: int = 0,
                         grouped: bool = False,
                         pre: ObservationTransform | None = None,
                         group_pre: GroupObservationsTransform | None = None,
                         use_world_bounds: bool = True,
                         of: int = 1,
                         select: int = 0,
                         **kwargs: Any) -> None:
    """
    Record the video from one episode of an environment.

    :param      env:               The environment
    :param      path:              Where to save the video
    :param      groups:            The configuration of the groups
    :param      policy:            The default policy
                                   (when not specified in the group config)
    :param      factor:            The real-time factor
    :param      seed:              The seed (only applies if ``of == 0``)
    :param      grouped:           Whether the policy is grouped
    :param      pre:               An optional transformation to apply to observations
                                   of all individual agents
    :param      group_pre:         An optional transformation to apply to observations
                                   of all groups
    :param      use_world_bounds:  Whether to keep the initial world bounds
    :param      select:            Which episode to select
                                   (ordered by reward, only applies if ``of > 0``)
    :param      of:                The number of runs to choose the episode from
    :param      kwargs:            Keywords arguments passed to the renderer

    """
    exp = make_experiment_with_env(env,
                                   groups=groups,
                                   policy=policy,
                                   record_reward=of > 0,
                                   grouped=grouped,
                                   pre=pre,
                                   group_pre=group_pre,
                                   record_success=False)
    record_run_video(exp,
                     path=path,
                     factor=factor,
                     seed=seed,
                     select=select,
                     of=of,
                     **kwargs)
