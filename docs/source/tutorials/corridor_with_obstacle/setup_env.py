import gymnasium as gym
from navground import sim
from navground.learning import ControlActionConfig, DefaultObservationConfig
from navground.learning.rewards import SocialReward
import navground.learning.scenarios

duration = 40.0
time_step = 0.1

action_config = ControlActionConfig(max_acceleration=1.0, max_angular_acceleration=10.0, 
                                    use_acceleration_action=True)

observation_config = DefaultObservationConfig(include_target_direction=True, include_velocity=True, 
                                              include_angular_speed=True, flat=False)

reward = SocialReward(safety_margin=0.04)

with open('sensor.yaml') as f:
    sensor = sim.load_state_estimation(f.read())

with open('scenario.yaml') as f:
    scenario = sim.load_scenario(f.read())
    
env = gym.make('navground', 
    scenario=scenario,
    sensor=sensor,
    action=action_config,
    observation=observation_config,
    time_step=time_step,
    max_duration=duration,
    reward=reward)
