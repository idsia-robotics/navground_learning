============
Introduction
============

This packages contains tools to train and use Machine Learning models inside a navground simulation.


Navground-Gymnasium integration 
===============================

Gymnasium
---------

`Gymnasium <https://gymnasium.farama.org>`_ by the Farama Foundation, which replaces the discontinued `Gym <https://www.gymlibrary.dev/index.html>`_ by OpenAI, is a Python package with a standardized API for reinforcement learning. The image and the example below are taken from `Gymnasium's docs <https://gymnasium.farama.org/content/basic_usage>`_.

At its core, Gymnasium implements the typical Markov Decision Process cycle of "observe → think → act → get reward":

.. figure:: https://gymnasium.farama.org/_images/AE_loop.png
   :width: 300

   The MDP cycle standardized by Gymnasium environments 

#. the process is initialized

#. and updated iteratively:

   * the agent applies an action
  
   * and gets observation and reward

#. until the process ends

which, using the API, is typically implemented like

.. code-block:: python
   
   import gymnasium as gym
   
   environment = gym.make("MyEnviroment")
   observation, info = environment.reset()
   
   for _ in range(1000):
       action = evaluate_my_policy(observation)
       observation, reward, terminated, truncated, info = environment.step(action)
   
       if terminated or truncated:
           observation, info = env.reset()
   
   env.close()


Environments include everything needed to run specific simulations. There are several available, ranging from Atari-like games to 3D robotics simulations.

PettingZoo
----------

`PettingZoo <https://pettingzoo.farama.org>`_ also by the Farama Foundation, extends Gymnasium to multi-agent system. It support two API: 

- `AEC <https://pettingzoo.farama.org/api/aec>`_, in which agents act one at the time, like in turn-based games;
- `Parallel <https://pettingzoo.farama.org/api/parallel>`_, in which agents at the same time and is more suitable to interact with navground.

The Parallel API is similar to Gymnasium, with the difference that actions, rewards, observations, ..., are indexed by an agent identifier:

.. code-block:: python
   
   import pettingzoo as pz
   
   environment = MyMultiAgentEnviroment()
   observations, infos = environment.reset()
   
   for _ in range(1000):
       actions = {index: evaluate_my_policy(observation) 
                  for index, observation in observations.items()}
       observations, rewards, terminations, truncations, infos = environment.step(actions)

       # Instead of looking at terminations and truncations
       # we can directly check that there are still some agents alive
       if not env.agents:
            break
      
   env.close()


.. note::

   We can convert between environments with AEC and Parallel API using
   `convertion wrappers <https://pettingzoo.farama.org/api/wrappers/pz_wrappers/#conversion-wrappers>`_.

   Moreover, we can convert PettingZoo environments in which all agents share the same action and observation spaces to 
   a vectorized Gymnasium environment that concatenate all the actions, observations and other infos using  `SuperSuit wrappers <https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/vector_constructors.py>`_. This way, we can use ML libraries that works with Gymanasium to train distributed multi-agent systems.


Navground
---------

`Navground <https://idsia-robotics.github.io/navground/_build/html/index.html>`_ simulations have a similar cycle where

#. a world is initialized from a *scenario*
#. and updated iteratively:

   * agents update the environment *state estimation* (i.e., using a sensor)
   * agents compute and actuate a control commands from a *navigation behavior*
   * the physics is updated resolving collisions

#. until the simulation ends

The API is slightly simpler

.. code-block:: python

   import navground as sim
   
   scenario = MyScenario(...)
   world = scenario.make_world(seed=1)
   for _ in range(1000):
       world.update(time_step=0.1)

as :py:meth:`navground.sim.World.update` groups together the steps of policy (in navground *behavior*) evaluation and actuation.

Navground simulations may features many diverse agents types, using different kinematics, tasks, state estimation, and navigation behaviors.

.. note::

   Navground is extensible. You can implement new behaviors, tasks, scenarios and in particular new sensor models, in C++ or in Python.
   Therefore, we can build a custom navigation environment for machine learning, where agents use a particular sensing models, by sub-classing :py:class:`navground.sim.Sensor`.

Navground Gymnasium Environment
-------------------------------

.. currentmodule:: navground.learning

:py:class:`.env.NavgroundEnv` wraps a :py:class:`navground.sim.Scenario` in an :py:class:`gymnasium.Env` that conforms to the standard API expected by gymnasium, with actions and observations linked to a *single* navground agent. In particular (with some simplifications):

.. code-block:: python

   NavgroundEnv(scenario: sim.Scenario, 
                action_config: ActionConfig, 
                observation_config: ObservationConfig, 
                sensor: sim.Sensor | None = None, ...)
  
instantiates a gymnasium environment whose worlds will be spawned using a navground scenario. If specified, the agent will use a sensor to generate observations, instead of its predefined state estimation. The action and observation spaces of the agent can be customized, for instance whether to include the distance to the target, or to control the agent orientation.
  
.. code-block:: python
 
   NavgroundEnv.reset(seed: int | None = None, options: dict[str, Any] | None = None)

Initializes a navground world from the navground scenario using a random seed and selects one of navground agents.

.. code-block:: python
  
   NavgroundEnv.step(action: numpy.typing.NDArray[Any])

Passes the action to the selected navground agent, updates the navground world and return the selected navground agent's observations and reward.


By specifying 

- :py:class:`.ObservationConfig`, we control how to
  convert a :py:class:`navground.core.SensingState` to gymnasium observations. As of now, as single concrete class is implemented:

  - :py:class:`.DefaultObservationConfig` that configures which parts of the ego-state and target information to include in the observations.

- :py:class:`.ActionConfig`, we control how to
  convert gymnasium actions to :py:class:`navground.core.Twist2` to be actuated by a  :py:class:`navground.core.Behavior`, with different subclasses:

  - :py:class:`.ControlActionConfig` where the policy outputs a control command
  - :py:class:`.ModulationActionConfig` where the policy outputs parameters of an underlying deterministic navigation behavior.

PettingZoo Navground Environment
--------------------------------

Similarly, :py:class:`.parallel_env.MultiAgentNavgroundEnv` provides a environment for which actions and observations are linked to a *multiple* navground agents.

:py:func:`.parallel_env.parallel_env` instantiate an environment where different agents may use different configurations (such as action spaces, rewards, ...), while
:py:func:`.parallel_env.shared_parallel_env` instantiate an environment where all specified agents share the same configuration.

The rest of the functionality is very similar to the Gymnasium Environment (and in fact, they share the same base class), but conform to the PettingZoo API instead.


Train ML policies in navground 
==============================

.. note::

   Have a look at the tutorials to see the interaction between gymnasium and navground in action and how to use it to train a navigation policy using IL or RL.

Imitation Learning
------------------

Using the navground environments, we can train a policy that imitates
one of the navigation behaviors implemented in navground, using any of the available sensors. 

We include helper classes that wraps the Python package `imitation <https://imitation.readthedocs.io/en/latest/>`_ by the Center for Human-Compatible AI
to offer simplified interface, yet nothing prevent to use the original API.

To learn to imitate a behavior, we can run


.. code-block:: python

   import gymnasium as gym
   import navground.learning.env
   from navground.learning.il import BC, DAgger

   env = gym.make("navground", scenario=..., sensor=...,
                  observation_config=..., action_config=..., 
                  max_episode_steps=100)

   # Behavior cloning
   bc = BC(env=env, runs=100)
   bc.learn(n_epochs=1)
   bc.save("BC")

   # DAgger
   dagger = DAgger(env=env)
   dagger.learn(total_timesteps=10_000, 
                rollout_round_min_timesteps=100)
   dagger.save("DAgger")

Reinforcement Learning
----------------------

Using the navground-gymnasium environment, we can train a policy to navigate among other agents controlled by navground, for instance using the RL algorithm implemented in `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_ by 
DLR-RM.

.. code-block:: python

   import gymnasium as gym
   import navground.learning.env
   from stable_baselines3 import SAC

   env = gym.make("navground", scenario=..., sensor=...,
                  observation_config=..., action_config=..., 
                  max_episode_steps=100)
   sac = SAC("MlpPolicy", env)
   sac.learn(total_timesteps=10_000)
   sac.save("SAC")

Parallel Multi-agent Learning
-----------------------------

Using the multi-agent navground-gymnasium environment, we can train a policy in parallel for all agents in the environment, that is, the agents
learn to navigate among peers that are learning the *same* policy.
We instantiate the parallel environment using :py:func:`.parallel_env.shared_parallel_env`, and transform it to a Stable-Baseline compatible (sigle-agent) vectorized environment using :py:func:`.parallel_env.make_vec_from_penv`. While learning, from the view-point of the ``SAC`` algorithm, rollouts will generate by a single agent in ``n`` environments that compose ``venv``, while in reality they will be generate in a single ``penv`` by ``n`` agents.

.. code-block:: python

   import gymnasium as gym
   from navground.learning.parallel_env import make_vec_from_penv, shared_parallel_env
   from stable_baselines3 import SAC

   penv = shared_parallel_env(scenario=..., sensor=...,
                              observation_config=..., action_config=..., 
                              max_episode_steps=100)
   venv = make_vec_from_penv(penv)
   psac = SAC("MlpPolicy", venv)
   psac_ma.learn(total_timesteps=10_000)
   psac.save("PSAC")


Evaluation
==========

Once trained, we can evaluate the policies with common tools, such as 
:py:func:`stable_baselines3.common.evaluation.evaluate_policy` and its extensions in :py:mod:`.evaluation` that supports parallel environments with groups using different policies.


Use ML policies in navground 
----------------------------

Evaluation can also be performed using the tools available in navground,
which are specifically designed to support large experiments with many runs and agents, distributing the work over multiple processor if desired.

Once we have trained a policy (and possibly exported it to onnx using :py:func:`.onnx.export`), :py:class:`.behaviors.PolicyBehavior` executes it as a navigation behavior in navground. As a basic example, we can load it and assign it to some of the agents in the simulation:

.. code-block:: python

   import navground as sim
   import gymnasium as gym
   from navground.learning.behaviors import PolicyBehavior

   # we load the same scenario and sensor used to train the policy
   scenario = sim.Scenario.load(...)
   sensor = sim.Sensor.load(...)
   world = scenario.make_world(seed=1)
   
   # and configure the first five agents to use the policy
   # instead of the original behavior
   for agent in world.agents[:5]:
      agent.behavior = PolicyBehavior.clone_behavior(
         agent.behavior, policy='policy.onnx', 
         action_config=..., observation_config=...)
      agent.state_estimation = sensor
   
   world.run(time_step=0.1, steps=1000)

In practice, we do not need to perform the configuration manually. Instead, we can load it from a YAML file (exported e.g. using :py:func:`.io.save_as_behavior`), like common in navground:

.. code-block:: YAML
   :caption: scenario.yaml

   groups:
     - number: 5
       behavior:
         type: PolicyBehavior
         policy_path: policy.onnx
         # action and observation config
         ...
       state_estimation:
         # sensor config
         ...
       # remaining of the agents config
       ...

When loaded, the 5 agents in this group will use the policy to navigate

.. code-block:: python

   import navground as sim

   # loads the navground.learning components such as PolicyBehavior
   sim.load_plugins()

   with open('scenario.yaml') as f:
      scenario = sim.Scenario.load(f.read())

   world = scenario.make_world(seed=1)
   world.run(time_step=0.1, steps=1000)

or we could embed it in an experiment to record trajectories and performance metrics:

.. code-block:: YAML
  :caption: experiment.yaml

   runs: 1000
   time_step: 0.1
   steps: 10000
   record_pose: true
   record_efficacy: true
   scenario:
       groups:
         - number: 5
           behavior:
             type: PolicyBehavior
             policy_path: policy.onnx
           ...

Acknowledgement and disclaimer
==============================

The work was supported in part by `REXASI-PRO <https://rexasi-pro.spindoxlabs.com>`_ H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement no. 101070028.

.. image:: https://rexasi-pro.spindoxlabs.com/wp-content/uploads/2023/01/Bianco-Viola-Moderno-Minimalista-Logo-e1675187551324.png
  :width: 300
  :alt: REXASI-PRO logo

The work has been partially funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.
