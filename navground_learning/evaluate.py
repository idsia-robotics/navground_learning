import dataclasses as dc
from functools import partial
from typing import Any
import statistics

import numpy as np
from navground import sim

from .config import Indices, WorldConfig, GroupConfig
from .env.base import NavgroundBaseEnv
from .probes.reward import RewardProbe
from .reward import Reward
from .scenarios.evaluation import EvaluationScenario


def make_experiment(scenario: sim.Scenario,
                    config: WorldConfig = WorldConfig(),
                    max_duration: float = -1.0,
                    steps: int = 1000,
                    time_step: float = 0.1,
                    bounds: tuple[np.ndarray, np.ndarray] | None = None,
                    terminate_outside_bounds: bool = False,
                    seed: int = 0,
                    terminate_when_all_idle_or_stuck: bool = True,
                    deterministic: bool = True) -> sim.Experiment:
    if max_duration > 0:
        steps = int(max_duration / time_step)
    experiment = sim.Experiment(time_step=time_step, steps=steps)
    experiment.terminate_when_all_idle_or_stuck = terminate_when_all_idle_or_stuck
    if config:
        experiment.scenario = EvaluationScenario(
            config=config,
            scenario=scenario,
            bounds=bounds,
            terminate_outside_bounds=terminate_outside_bounds,
            deterministic=deterministic)
    else:
        experiment.scenario = scenario
    experiment.run_index = seed
    experiment.add_record_probe("reward", partial(RewardProbe, config=config))
    return experiment


def make_experiment_with_env(env: NavgroundBaseEnv,
                             policies: list[tuple[Indices, Any]] = [],
                             policy: Any = None,
                             reward: Reward | None = None,
                             use_first_reward: bool = True,
                             steps: int = 1000,
                             seed: int = 0) -> sim.Experiment:
    if not env._scenario:
        raise ValueError("No scenario")
    if reward is None and use_first_reward:
        reward = env.config.get_first_reward()
    if policy is not None:
        policies = policies[:] + [(None, policy)]
    config = dc.replace(env.config, policies=policies, reward=reward)
    return make_experiment(
        scenario=env._scenario,
        config=config,
        max_duration=env.max_duration,
        steps=steps,
        time_step=env.time_step,
        bounds=env.bounds,
        terminate_outside_bounds=env.terminate_outside_bounds,
        seed=seed)


# returns an array of the sum of agent rewards during runs
def _evaluate_expert(scenario: sim.Scenario,
                     config: WorldConfig = WorldConfig(),
                     runs: int = 1,
                     **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    exp = make_experiment(scenario=scenario, config=config, **kwargs)
    exp.number_of_runs = runs
    exp.run()
    rs = [np.asarray(run.get_record("reward")) for run in exp.runs.values()]
    rewards = np.concatenate([np.sum(r, axis=0) for r in rs])
    lengths = np.concatenate([[r.shape[0]] * r.shape[1] for r in rs])
    return rewards, lengths


# returns an array of the sum of agent rewards during runs
def evaluate_expert(scenario: sim.Scenario,
                    reward: Reward,
                    indices: Indices = None,
                    runs: int = 1,
                    **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    config = WorldConfig(groups=[GroupConfig(reward=reward, indices=indices)])
    return _evaluate_expert(scenario, config, runs=runs, **kwargs)


# Mimic SB3 `evaluate_policy`
# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#module-stable_baselines3.common.evaluation
def evaluate_expert_with_env(
        env: NavgroundBaseEnv,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        return_episode_rewards: bool = False,
        **kwargs: Any) -> tuple[list[float], list[int]] | tuple[float, float]:
    if not env._scenario:
        raise ValueError("No scenario")
    config = WorldConfig(groups=[
        GroupConfig(reward=g.reward, indices=g.indices)
        for g in env.config.groups
    ])
    rs, ls = _evaluate_expert(scenario=env._scenario,
                              config=config,
                              runs=n_eval_episodes,
                              deterministic=deterministic,
                              max_duration=env.max_duration,
                              time_step=env.time_step,
                              **kwargs)
    if not return_episode_rewards:
        return statistics.mean(rs), statistics.stdev(rs)
    return rs.tolist(), ls.tolist()
