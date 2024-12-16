# navground_learning

A Python package that adds Imitation or Reinforcement Learning to the tools user can play with in the **Nav**igation Play**ground** [navground](https://github.com/idsia-robotics/navground).

The package provides configurable single-agent [Gymnasium](https://gymnasium.farama.org/index.html)  and multi-agent Parallel [PettingZoo](https://pettingzoo.farama.org/index.html) environments. Users, by combining navigation scenarios and sensors, can build any kind of environment where to learn navigation policies.

## Main features

- Single-agent Gymnasium environment that expose actions, obervations and rewards of a single agent in a navground scenario.
- Multi-agent PettingZoo environment that expose actions, observations and rewards of multiple agents, possibly heterogenous, in a navground scenario.
- Ability to execute a navigation policy in navground, with support for [ONNX](https://onnx.ai) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io).
- User-extendable configuration for translating between navground commands/states and the POMDP actions/observations.
- Reward functions for navigation tasks.
- Import/export from/to YAML, like common in navground.
- Tools to evaluate the policies during and after training, inside and outside of the environments.

## Installation

```
pip install navground_learning[all]
```

We support Python>=3.10. Users should prefer Python<=3.12 because important third-party packages like Stable-BaseLine3 or PyTorch do not support Python3.13 yet.

## Example

Note that to run the example, you need to install the `rl` and `inference` optional dependecies:

```
pip install navground_learning[rl,inference]
```

which are installed if you installed `navground_learning[all]`.


The example instantiate the environment of [one of the tutorials](https://idsia-robotics.github.io/navground_learning/tutorials/corridor_with_obstacle.html), in which it trains a policy using `SAC`.
Then, it export the policy as an `ONNX`, which it uses to run a navground experiment, recording the trajectories to an HDF5 file.

```python
from navground.learning import evaluation, io
from navground.learning.examples import corridor_with_obstacle
from stable_baselines3 import SAC

# Creates a Gymnasium environment 
env = corridor_with_obstacle.get_env()
# Train using Stable-Baseline3
model = SAC("MlpPolicy", env).learn(total_timesteps=1_000)
# Export the model to ONNX
io.export_behavior(model, "model")

# Evaluate the policy with a navground experiment
exp = evaluation.make_experiment_with_env(env, policy="model/policy.onnx")
exp.number_of_runs = 100
exp.record_config.pose = True
exp.save_directory = '.'
exp.run()
```
## Documentation

The [project documentation](https://idsia-robotics.github.io/navground_learning) in addition to the API reference, includes 
several [tutorials](https://idsia-robotics.github.io/navground_learning/tutorials) of Machine-Learning policies in navigation, ranging from very basics to complex multi-agent scenarios. 

## License and copyright

This software is free for reuse according to the attached MIT license.

## Acknowledgement and disclaimer

The work was supported in part by [REXASI-PRO](https://rexasi-pro.spindoxlabs.com) H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement no. 101070028.

<img src="https://rexasi-pro.spindoxlabs.com/wp-content/uploads/2023/01/Bianco-Viola-Moderno-Minimalista-Logo-e1675187551324.png"  width="300">

The work has been partially funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.