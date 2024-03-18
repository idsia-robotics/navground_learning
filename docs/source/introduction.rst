============
Introduction
============

This packages contains tools to train and use Machine Learning models inside a navground simulation.


Navground-Gymnasium integration 
===============================

Gymnasium
---------

`Gymnasium <https://gymnasium.farama.org>`_ by the Farama Foundation, which replaces the discontinued `Gym <https://www.gymlibrary.dev/index.html>`_ by OpenAI, is a Python package with a standardized API for reinforcement learning. The image and example are taken from `Gymnasium's docs <https://gymnasium.farama.org/content/basic_usage>`_.

At it's core, Gymnasium implements the typical Markov Decision Process cycle of "observe → think → act → get reward":

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
   
   world = sim.World()
   scenario = MyScenario(...)
   scenario.init_world(world)
   
   for _ in range(1000):
       world.update(time_step=0.1)

as :py:meth:`navground.sim.World.update` groups together the steps of policy (in navground *behavior*) evaluation and actuation.

Navground simulations may features many diverse agents types, using different kinematics, tasks, state estimation, and navigation behaviors.

.. note::

   Navground is extensible. You can implement new behaviors, tasks, scenarios and in particular new sensor models, in C++ or in Python.
   Therefore, you can build a custom navigation environment for machine learning, where agents use a particular sensing models, by subclassing :py:class:`navground.sim.Sensor`.

Navground Gymnasium Environment
-------------------------------

:py:class:`navground_learning.env.NavgroundEnv` wraps a :py:class:`navground.sim.Scenario` in an :py:class:`gymnasium.Env` that conforms to the standard API expected by gymnasium, with actions and observations linked to a *single* navground agent. In particular (with some simplifications):

.. code-block:: python

   NavgroundEnv(scenario: sim.Scenario, config: GymAgentConfig = GymAgentConfig(), sensor: sim.Sensor | None = None, ...)
  
Instantiates a gymnasium environment whose worlds will be spawned using a navground scenario. If specified, the agent will use a sensor to generate observations, instead of its predefined state estimation. The action and observation spaces of the agent can be customized, for instance whether to include the distance to the target, or to control the agent orientation.
  
.. code-block:: python
 
   NavgroundEnv.reset(seed: int | None = None, options : Dict | None = None)

Initializes a navground world from the navground scenario using a random seed and selects one of navground agents.

.. code-block:: python
  
   NavgroundEnv.step(action: numpy.ndarray)

Passes the action to the selected navground agent, updates the navground world and return the selected navground agent's observations and reward.


By specifying a :py:class:`navground_learning.GymAgentConfig`, we control how to

* convert a :py:class:`navground.core.SensingState` to gymnasium observations
* convert gymnasium actions to :py:class:`navground.core.Twist` to be actuated by a  :py:class:`navground.core.Behavior`


Use ML policies in navground 
============================

Once we have trained a policy in the navground-gymansium enviroment,
the class :py:class:`navground_learning.behaviors.PolicyBehavior`
integrates it in navground. You can exchange the original behavior of an agent to use a policy instead, like

.. code-block:: python

   import navground as sim
   import gymnasium as gym
   from navground_learning.behaviors import PolicyBehavior

   scenario = MyScenario(...)
   sensor = MySensor(....)
   env = gym.make("navground", scenario=scenario, sensor=sensor)

   # train a policy using the scenario and sensor
   # ...
   # evaluate the policy

   world = sim.World()
   scenario.init_world(world)
   
   # set the first agent to use the policy and the sensor, 
   # instead of the original behavior and state estimation

   world.agents[0].behavior = PolicyBehavior.clone_behavior(
      agent.behavior, policy=my_trained_policy, config=my_config)
   world.agents[0].state_estimation = my_sensor
   
   for _ in range(1000):
       world.update(time_step=0.1)

or you can directly configure the scenario (for instance in YAML) so that some agents uses this policy:

.. code-block:: python

   import navground as sim
   from navground_learning.behaviors import PolicyBehavior

   world = sim.World()
   scenario = sim.load_scenario(...)
   scenario.init_world(world)

   for _ in range(1000):
      world.update(time_step=0.1)

.. note:: 

   By using :py:class:`navground_learning.behaviors.PolicyBehavior`, we don't need to run the gymnasium environment anymore to perform validation simulation but can instead use the many tools available in navground. Nonetheless, we could use gymnasium for validation too, if we prefer it.
   


Train ML policies in navground 
==============================

.. note::

   Have a look at the tutorials to see the interaction between gymnasium and navground in action and how to use it to train a navigation policy using IL or RL.


Imitation Learning
------------------

Using the navground-gymnasium environment, we can train a policy that imitates
one of the navigation behaviors implemented in navground, using any of the available sensors. 

We include helper classes that wraps the Python package `imitation <https://imitation.readthedocs.io/en/latest/>`_ by the Center for Human-Compatible AI
to offer simplified interface, yet nothing prevent to use the original API.

To learn to imitate a behavior, we can run


.. code-block:: python

   import navground as sim
   import gymnasium as gym
   from navground_learning import il

   scenario = MyScenario(...)
   sensor = MySensor(....)
   env = gym.make("navground", scenario=scenario, sensor=sensor, max_episode_steps=1000)

   # Behavior cloning
   trainer = il.bc.Trainer(env=env, runs=100)
   trainer.train(n_epochs=1)

   # DAgger
   # trainer = il.dagger.Trainer(env=env)
   # trainer.train(n_epochs=1)

   trainer.save("results")
   behavior = trainer.make_behavior()

   # use the behavior in navground
   # ...


Reinforcement  Learning
-----------------------

Using the navground-gymnasium environment, we can train a policy to navigate among other agents controlled by navground, for instance using the RL algorithm implemented in `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_ by 
DLR-RM.

.. code-block:: python

   import navground as sim
   import gymnasium as gym
   from stable_baselines3 import SAC

   scenario = MyScenario(...)
   sensor = MySensor(....)
   env = gym.make(
      "navground", scenario=scenario, 
      sensor=sensor,max_episode_steps=1000)

   model = SAC("MlpPolicy", env, verbose=0)
   model.learn(total_timesteps=100000, progress_bar=True);


Acknowledgement and disclaimer
==============================

The work was supported in part by `REXASI-PRO <https://rexasi-pro.spindoxlabs.com>`_ H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement no. 101070028.

.. image:: https://rexasi-pro.spindoxlabs.com/wp-content/uploads/2023/01/Bianco-Viola-Moderno-Minimalista-Logo-e1675187551324.png
  :width: 300
  :alt: REXASI-PRO logo

The work has been partially funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.
