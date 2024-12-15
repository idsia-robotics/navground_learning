import gymnasium as gym
import navground.learning.scenarios  # noqa
from navground import sim
from navground.learning import ControlActionConfig, DefaultObservationConfig
from navground.learning.env import BaseEnv
from navground.learning.rewards import SocialReward


def get_env(flat: bool = True,
            duration: float = 40.0,
            time_step: float = 0.1) -> BaseEnv:

    scenario = sim.load_scenario("""
type: CorridorWithObstacle
length: 1.0
width: 1.0
min_radius: 0.2
max_radius: 0.2
groups:
  -
    type: thymio
    number: 1
    radius: 0.08
    control_period: 0.05
    color: blue
    kinematics:
      type: 2WDiff
      wheel_axis: 0.094
      max_speed: 0.12
    behavior:
      type: HL
      optimal_speed: 0.12
      horizon: 10
      tau: 0.25
      eta: 0.5
      safety_margin: 0.05
      barrier_angle: 1.0
    state_estimation:
      type: Bounded
      range: 1.0
      update_static_obstacles: true
""")

    sensor = sim.load_state_estimation("""
type: Combination
sensors:
  - type: Boundary
    min_y: 0
    max_y: 1
    range: 1
  - type: Discs
    number: 1
    range: 1
    max_speed: 0.0
    max_radius: 0.0
    include_valid: false
""")

    action_config = ControlActionConfig(max_acceleration=1.0,
                                        max_angular_acceleration=10.0,
                                        use_acceleration_action=True)

    observation_config = DefaultObservationConfig(include_target_distance=False,
                                                  include_target_direction=True,
                                                  include_velocity=True,
                                                  include_angular_speed=True,
                                                  flat=flat)
    return gym.make('navground',
                    scenario=scenario,
                    sensor=sensor,
                    action=action_config,
                    observation=observation_config,
                    time_step=time_step,
                    max_duration=duration,
                    reward=SocialReward(safety_margin=0.04))
