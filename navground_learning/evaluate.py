import dataclasses as dc
from functools import partial
from typing import Any

import numpy as np
from navground import sim

from .config import Indices, WorldConfig
from .env.base import NavgroundBaseEnv
from .probes.reward import RewardProbe
from .reward import Reward
from .scenarios.evaluation import EvaluationScenario


def make_experiment(
        scenario: sim.Scenario,
        config: WorldConfig = WorldConfig(),
        max_duration: float = -1.0,
        steps: int = 1000,
        time_step: float = 0.1,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        terminate_outside_bounds: bool = False,
        seed: int = 0,
        terminate_when_all_idle_or_stuck: bool = True) -> sim.Experiment:
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
        )
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


# Mimic SB3 `evaluate_policy`
def evaluate_expert(scenario: sim.Scenario,
                    reward: Reward,
                    runs: int = 1,
                    seed: int = 0,
                    **kwargs: Any) -> tuple[list[float], list[int]]:
    exp = make_experiment(scenario=scenario,
                          config=WorldConfig(reward=reward),
                          seed=seed,
                          **kwargs)
    exp.number_of_runs = runs
    exp.run()
    rewards = [
        np.mean(np.sum(np.asarray(run.get_record("reward")), axis=0))
        for run in exp.runs.values()
    ]
    lengths = [
        len(np.asarray(run.get_record("reward"))) for run in exp.runs.values()
    ]
    return rewards, lengths


def evaluate_expert_with_env(env,
                             reward: Reward | None = None,
                             **kwargs: Any) -> tuple[list[float], list[int]]:
    if not env._scenario:
        raise ValueError("No scenario")
    if reward is None:
        reward = env.config.get_first_reward()
    return evaluate_expert(scenario=env._scenario, reward=reward, **kwargs)
