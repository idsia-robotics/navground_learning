from __future__ import annotations

import pathlib as pl
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

try:
    from contextlib import chdir  # Python>=3.11
except ImportError:
    import contextlib
    import os

    @contextlib.contextmanager  # type: ignore[no-redef]
    def chdir(path: os.PathLike[str]) -> Iterator[None]:
        _old = os.getcwd()
        os.chdir(os.path.abspath(path))
        try:
            yield
        finally:
            os.chdir(_old)

import yaml
from navground import core, sim

from .behaviors.policy import PolicyBehavior
from .config import ControlActionConfig, DefaultObservationConfig
from .env import BaseEnv, NavgroundEnv
from .internal.base_env import NavgroundBaseEnv
from .internal.group import Agent
from .onnx.export import export
from .parallel_env import BaseParallelEnv, MultiAgentNavgroundEnv
from .types import PathLike

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import BasePolicy

    from .il import BaseILAlgorithm


def _get_agent(env: NavgroundBaseEnv,
               index: int | None = None) -> Agent | None:
    if index is None:
        if env._possible_agents:
            return next(iter(env._possible_agents.values()))
        return None
    return env._possible_agents.get(index, None)


def _get_policy_behavior(env: NavgroundBaseEnv,
                         index: int | None = None) -> PolicyBehavior | None:

    agent = _get_agent(env, index)
    if (agent and agent.navground and agent.gym and agent.navground.behavior
            and isinstance(agent.gym.action_config, ControlActionConfig)
            and isinstance(agent.gym.observation_config,
                           DefaultObservationConfig)):
        return PolicyBehavior.clone_behavior(
            behavior=agent.navground.behavior,
            action_config=agent.gym.action_config,
            observation_config=agent.gym.observation_config,
            policy=None)
    return None


def _get_sensor(env: NavgroundBaseEnv,
                index: int | None = None) -> sim.Sensor | None:
    agent = _get_agent(env, index)
    if agent:
        return agent.sensor
    return None


def save_env(
    env: BaseEnv | BaseParallelEnv,
    path: PathLike,
) -> None:
    """
    Export a NavgroundEnv to YAML

    :param      path:    The directory where to save the files
    :param      env:     The environment.
    :raises TypeError:   If ``env.unwrapped`` is not of a navground environment
    """
    if not isinstance(env.unwrapped, NavgroundBaseEnv):
        raise TypeError("Not a navground environment")
    with open(path, 'w') as f:
        f.write(yaml.safe_dump(env.unwrapped.asdict))


def load_env(path: PathLike) -> MultiAgentNavgroundEnv | NavgroundEnv:
    with open(path) as f:
        data = yaml.safe_load(f.read())
    return cast(MultiAgentNavgroundEnv | NavgroundEnv,
                NavgroundBaseEnv.from_dict(data))


def export_behavior(
    model: BaseAlgorithm | BaseILAlgorithm,
    path: PathLike,
) -> None:
    """
    Export a model using :py:func:`export_policy_as_behavior`

    :param      model:   The model
    :param      path:    The directory where to save the files
    """
    venv = model.env
    if venv:
        value = venv.get_attr("asdict", [0])[0]
        env: BaseEnv | BaseParallelEnv | None = cast(
            BaseEnv | BaseParallelEnv, NavgroundBaseEnv.from_dict(value))
    else:
        env = None
    export_policy_as_behavior(path=path, policy=model.policy, env=env)


def export_policy_as_behavior(path: PathLike,
                              policy: BasePolicy | None = None,
                              env: BaseEnv | BaseParallelEnv | None = None,
                              index: int | None = None) -> None:
    """
    Export a policy (using :py:func:`navground.learning.onnx.export`)
    together with the YAML representation of the behavior and sensor to use it.

    The behavior and sensor refer to an agent at a given index in the environment.
    If they cannot be retrieved, the related files will not be saved.

    :param      path:    The directory where to save the files
    :param      policy:  The policy
    :param      env:     The environment.
    :param      index:   The index of the agent in the environment.
                         If not provided, selects one of the possible agents.
    """
    path = pl.Path(path)
    if path.is_file():
        warnings.warn(f"{path} points to a file", stacklevel=1)
        return
    if not path.exists():
        path.mkdir(parents=True)
    if policy:
        export(policy, path / "policy.onnx")
    if env and not isinstance(env, NavgroundBaseEnv):
        env = env.unwrapped
    if not (env and isinstance(env, NavgroundBaseEnv)):
        return
    behavior = _get_policy_behavior(env, index=index)
    if behavior:
        if policy:
            behavior.set_policy_path('policy.onnx', load_policy=False)
        with open(path / 'behavior.yaml', 'w') as f:
            f.write(behavior.dump())
    sensor = _get_sensor(env, index=index)
    if sensor:
        with open(path / 'sensor.yaml', 'w') as f:
            f.write(sensor.dump())
    # TODO(Jerome): Maybe add a scenario


def load_behavior(
    path: PathLike
) -> tuple[PolicyBehavior | None, sim.StateEstimation | None]:
    """
    Load behavior and sensor previously saved in a directory
    using :py:func:`export_policy_as_behavior`.

    :param      path:  The directory path

    :returns:   A policy behavior and a sensor, set to None if the could not be loaded.
    """
    path = pl.Path(path)
    behavior: PolicyBehavior | None = None
    sensor: sim.StateEstimation | None = None
    with chdir(path):
        with open('behavior.yaml') as f:
            b = core.Behavior.load(f.read())
            if isinstance(b, PolicyBehavior):
                behavior = b
        with open('sensor.yaml') as f:
            sensor = sim.StateEstimation.load(f.read())
    return behavior, sensor
