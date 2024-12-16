from typing import Any

import gymnasium as gym
from navground import sim
from navground.learning import ControlActionConfig, DefaultObservationConfig
from navground.learning.env import BaseEnv
from navground.learning.parallel_env import BaseParallelEnv, shared_parallel_env
from navground.learning.rewards import SocialReward


def get_env(flat: bool = True,
            use_acceleration_action: bool = True,
            multi_agent: bool = False,
            **kwargs: Any) -> BaseEnv | BaseParallelEnv:
    """
    Creates the an environment where 20 agents travel back and forth
    between way-points, crossing in the middle.

    :param flat: Whether the observation space is flat
    :param use_acceleration_action: Whether actions are acceleration or velocities
    :param multi_agent: Whether to expose all agents or just one.
    :param kwargs: Arguments passed to the environment constructor

    :returns: A Parallel PettingZoo environment if `multi_agent` is set,
        else a Gymnasium environment.

    """

    scenario = sim.load_scenario("""
type: Cross
agent_margin: 0.1
side: 4
target_margin: 0.1
tolerance: 0.5
groups:
  -
    type: thymio
    number: 20
    radius: 0.1
    control_period: 0.1
    speed_tolerance: 0.02
    color: gray
    kinematics:
      type: 2WDiff
      wheel_axis: 0.094
      max_speed: 0.12
    behavior:
      type: HL
      optimal_speed: 0.12
      horizon: 5.0
      tau: 0.25
      eta: 0.5
      safety_margin: 0.1
    state_estimation:
      type: Bounded
      range: 5.0
""")

    sensor = sim.load_state_estimation("""
type: Discs
number: 5
range: 5.0
max_speed: 0.12
max_radius: 0.1
""")

    observation_config = DefaultObservationConfig(
        flat=flat, include_target_direction=True, include_target_distance=True)
    action_config = ControlActionConfig(
        use_acceleration_action=use_acceleration_action)
    if use_acceleration_action:
        observation_config.include_angular_speed = True
        observation_config.include_velocity = True
        action_config.max_acceleration = 1.0
        action_config.max_angular_acceleration = 10.0
    vs: dict[str, Any] = dict(scenario=scenario,
                              sensor=sensor,
                              action=action_config,
                              observation=observation_config,
                              reward=SocialReward(),
                              time_step=0.1)
    vs.update(kwargs)
    if not multi_agent:
        return gym.make('navground', **vs)
    return shared_parallel_env(**vs)
