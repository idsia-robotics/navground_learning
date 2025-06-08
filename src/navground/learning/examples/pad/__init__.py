from __future__ import annotations

import dataclasses as dc
from typing import Any, cast

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from navground import sim

from ...config import ControlActionConfig, ObservationConfig, StateConfig
from ...env import BaseEnv
from ...parallel_env import BaseParallelEnv, shared_parallel_env
from ...rewards import EfficacyReward
from ...scenarios.pad import PadScenario, are_two_agents_on_the_pad
from ...state_estimations import CommSensor
from ...types import Reward, SensorSequenceLike


@dc.dataclass
class PadReward(EfficacyReward, register_name="Pad"):
    """
    An efficacy reward that also penalizes by ``pad_penalty`` when both
    agents are inside the pad area.

    When ``neighbor_weight > 0``, it includes the efficacy of the neighbor,
    weighted accordingly.
    """
    pad_penalty: float = 10
    neighbor_weight: float = 0

    def __call__(self, agent: sim.Agent, world: sim.World,
                 time_step: float) -> float:
        r = super().__call__(agent, world, time_step)
        if are_two_agents_on_the_pad(world, 0):
            r -= self.pad_penalty
        if self.neighbor_weight:
            r += self.neighbor_weight * sum(
                EfficacyReward.__call__(self, other, world, time_step)
                for other in world.agents if other is not agent)
        return r


def is_success(agent: sim.Agent, world: sim.World) -> bool:
    if agent.behavior and agent.behavior.target.direction is not None:
        return bool(
            np.dot(agent.position, agent.behavior.target.direction) >
            world.bounding_box.max_x)
    return False


def is_failure(agent: sim.Agent, world: sim.World) -> bool:
    return are_two_agents_on_the_pad(world)


def neighbor(
        range: float = 10,
        max_speed: float = 0.166
) -> sim.state_estimations.DiscsStateEstimation:
    """
    The sensors that detects the neighbor.

    :param      range:      The range
    :param      max_speed:  The neighbor maximum speed

    :returns:   The sensor
    """
    return sim.state_estimations.DiscsStateEstimation(number=1,
                                                      range=range,
                                                      max_speed=max_speed,
                                                      name="neighbor",
                                                      include_valid=False,
                                                      include_y=False,
                                                      use_nearest_point=False)


def marker(min_x: float = -1,
           max_x: float = 1) -> sim.state_estimations.MarkerStateEstimation:
    """
    The sensors that detects the pad.

    :param      min_x:  The lower bound of the relative (horizontal) position.
    :param      max_x:  The upper bound of the relative (horizontal) position.

    :returns:   The sensor
    """
    return sim.state_estimations.MarkerStateEstimation(
        reference_orientation=sim.state_estimations.MarkerStateEstimation.
        ReferenceOrientation.target_direction,
        name="pad",
        min_x=min_x,
        max_x=max_x,
        include_y=False)


def comm(size: int = 1,
         name: str = 'neighbor',
         binarize: bool = False) -> sim.Sensor:
    """
    The sensors that receives messages broadcasted by the neighbor.

    :param      size:      The size of the message
    :param      name:      The namespace
    :param      binarize:  Whether to binarize the message

    :returns:   The sensor
    """
    return CommSensor(binarize=binarize, size=size, name=name)


def get_env(action: ControlActionConfig,
            observation: ObservationConfig,
            sensors: SensorSequenceLike = tuple(),
            reward: Reward = PadReward(),
            max_duration: float = 20,
            time_step: float = 0.1,
            init_success: bool = False,
            intermediate_success: bool = False,
            include_success: bool = True,
            render_mode: str | None = None,
            render_kwargs: dict = {},
            state: StateConfig | None = None,
            multi_agent: bool = True,
            **kwargs: Any) -> BaseEnv | BaseParallelEnv:
    """
    Creates the an environment where 2 agents cross along a corridor where
    there is pad which should not be entered by more than one agent at the
    same time.

    :param      action:                The action config
    :param      observation:           The observation config
    :param      sensors:               The sensors
    :param      reward:                The reward function
    :param      max_duration:          The maximal duration [s]
    :param      time_step:             The time step [s]
    :param      init_success:          The initialization value for intermediate success
    :param      intermediate_success:  Whether to return intermediate success
    :param      include_success:       Whether to include success
    :param      render_mode:           The render mode
    :param      render_kwargs:         The rendering keywords arguments
    :param      state:                 The global state config
                                       (only relevant if ``multi_agent=True``)
    :param      multi_agent:           Whether the environments controls both agents
    :param      kwargs:                Keywords arguments passed to
                                       :py:class:`navground.learning.scenarios.PadScenario`.

    :returns: A Parallel PettingZoo environment if `multi_agent` is set,
        else a Gymnasium environment.
    """

    if not multi_agent:
        return gym.make('navground',
                        scenario=PadScenario(**kwargs),
                        action=action,
                        observation=observation,
                        reward=reward,
                        sensors=sensors,
                        time_step=time_step,
                        success_condition=is_success,
                        failure_condition=is_failure,
                        terminate_on_success=True,
                        terminate_on_failure=False,
                        init_success=init_success,
                        intermediate_success=intermediate_success,
                        include_success=include_success,
                        max_duration=max_duration)
    return shared_parallel_env(scenario=PadScenario(**kwargs),
                               action=action,
                               observation=observation,
                               reward=reward,
                               sensors=sensors,
                               time_step=time_step,
                               success_condition=is_success,
                               failure_condition=is_failure,
                               terminate_on_success=True,
                               terminate_on_failure=False,
                               max_duration=max_duration,
                               wait=True,
                               init_success=init_success,
                               include_success=include_success,
                               intermediate_success=intermediate_success,
                               render_mode=render_mode,
                               render_kwargs=render_kwargs,
                               state=state)


def plot_policy(policy: Any,
                cmap: str = 'RdYlGn',
                title='Distributed policy',
                speed: float = 0.01):
    xs = np.linspace(-1, 1, 101, dtype=np.float32)
    ys = np.linspace(-2, 2, 101, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    if isinstance(policy.observation_space, gym.spaces.Dict):
        obs = {
            'pad/x': xv,
            'neighbor/position': yv,
            'neighbor/velocity': np.full(101 * 101, speed, dtype=np.float32),
            'ego_velocity': np.full(101 * 101, speed, dtype=np.float32)
        }
        obs = {
            k: obs[k].reshape(-1,
                              *cast('gym.spaces.Box', v).shape)
            for k, v in policy.observation_space.items()
        }
    else:
        raise NotImplementedError
    act, _ = policy.predict(obs, deterministic=True)
    act = act.reshape(101, 101, -1)
    if isinstance(policy.action_space, gym.spaces.Discrete):
        act = act - 1
    elif isinstance(policy.action_space, gym.spaces.MultiBinary):
        act = act[..., 1] - act[..., 0]
    plt.figure(figsize=(6, 4))
    plt.imshow(act, vmin=-1, vmax=1, cmap=cmap)
    plt.xlabel('relative pad center position')
    plt.ylabel('neighbor position')
    plt.colorbar()
    plt.xticks(np.linspace(0, 101, 5),
               [f'{x:.2f}' for x in np.linspace(-1, 1, 5).round(2)])
    plt.yticks(np.linspace(0, 101, 5),
               [f'{x:.2f}' for x in np.linspace(-2, 2, 5).round(2)])
    plt.title(title)


def plot_policy_with_comm(policy: Any,
                          cmap: str = 'RdYlGn',
                          title='Distributed policy with comm',
                          speed: float = 0.01,
                          binarize: bool | None = None):
    xs = np.linspace(-1, 1, 101, dtype=np.float32)
    if isinstance(policy.observation_space, gym.spaces.Dict):
        if binarize is None:
            comm_space = policy.observation_space['neighbor/comm']
            binarize = not np.issubdtype(comm_space.dtype, np.floating)
        low = -1
        high = 1
        # low = comm_space.low[0]
        # high = comm_space.high[0]
        if not binarize:
            ys = np.linspace(low, high, 101, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            obs = {
                'pad/x': xv,
                'neighbor/comm': yv,
                'ego_velocity': np.full(101 * 101, speed, dtype=np.float32)
            }
            obs = {
                k: obs[k].reshape(-1,
                                  *cast('gym.spaces.Box', v).shape)
                for k, v in policy.observation_space.items()
            }
            act, _ = policy.predict(obs, deterministic=True)
            act = act.reshape(101, 101, -1)
            # if isinstance(policy.action_space, gym.spaces.Discrete):
            #     act = act - 1
            # elif isinstance(policy.action_space, gym.spaces.MultiBinary):
            #     act = act[..., 1] - act[..., 0]
            fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
            for i, (ax, t) in enumerate(
                    zip(axs, ('acceleration', 'tx'), strict=True)):
                im = ax.imshow(act[..., i], vmin=-1, vmax=1, cmap=cmap)
                ax.set_xlabel('relative pad center position')
                ax.set_ylabel('rx')
                ax.set_xticks(np.linspace(0, 101, 5),
                              np.linspace(-1, 1, 5).round(2))
                ax.set_yticks(np.linspace(0, 101, 5),
                              np.linspace(-1, 1, 5).round(2))
                ax.title.set_text(t)
            fig.colorbar(im, ax=axs.ravel().tolist())
            fig.suptitle(title)
        else:
            xv = np.concatenate([xs, xs], axis=-1)
            yv = np.array([0] * len(xs) + [1] * len(xs), dtype=np.float32)
            obs = {
                'pad/x': xv,
                'neighbor/comm': yv,
                'ego_velocity': np.full(len(xv), speed, dtype=np.float32)
            }
            obs = {
                k: obs[k].reshape(-1,
                                  *cast('gym.spaces.Box', v).shape)
                for k, v in policy.observation_space.items()
            }
            act, _ = policy.predict(obs, deterministic=True)
            if isinstance(policy.action_space, gym.spaces.Discrete):
                act = np.stack([(act % 3) - 1, act // 3], axis=-1)
            elif isinstance(policy.action_space, gym.spaces.MultiBinary):
                raise NotImplementedError
            act = act.reshape(2, 101, 2)
            fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
            for i, (ax, t) in enumerate(
                    zip(axs, ('acceleration', 'tx'), strict=True)):
                if i == 1:
                    ax.plot(xs, (act[0, ..., i] > 0).astype(int), label="RX=0")
                    ax.plot(xs, (act[1, ..., i] > 0).astype(int), label="RX=1")
                else:
                    ax.plot(xs, act[0, ..., i], label="RX=0")
                    ax.plot(xs, act[1, ..., i], label="RX=1")
                ax.set_xlabel('relative pad center position')
                ax.set_ylabel(t)
                ax.legend()
            fig.suptitle(title)
    else:
        raise NotImplementedError
