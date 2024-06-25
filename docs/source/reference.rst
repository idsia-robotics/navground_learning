=========
Reference
=========

Configuration
=============

Actions
~~~~~~~

End-to-end
----------

.. autoclass:: navground_learning.ControlActionConfig
   :members:

Behavior Modulation
-------------------

.. autoclass:: navground_learning.ModulationActionConfig
   :members:

Observations
~~~~~~~~~~~~

.. autoclass:: navground_learning.ObservationConfig
   :members:

Multi-agent system
~~~~~~~~~~~~~~~~~~

.. autoclass:: navground_learning.GroupConfig
   :members:

.. autoclass:: navground_learning.WorldConfig
   :members:

Reward functions
================

Null
~~~~

.. autofunction:: navground_learning.reward.NullReward

Social
~~~~~~

.. autofunction:: navground_learning.reward.SocialReward

Inference
=========

Behaviors
~~~~~~~~~

.. autoclass:: navground_learning.behaviors.PolicyBehavior
   :members:
   :show-inheritance:

Environments
============

Single-agent (Gymnasium)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: navground_learning.env.NavgroundEnv
   :members:

Multi-agent (Pettingzoo)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: navground_learning.env.pz.shared_parallel_env


.. autofunction:: navground_learning.env.pz.parallel_env

Learning
=========

Imitation Learning
~~~~~~~~~~~~~~~~~~

Behavior Cloning
----------------

.. autoclass:: navground_learning.il.bc.Trainer
   :members:
   :inherited-members:

DAgger
------

.. autoclass:: navground_learning.il.dagger.Trainer
   :members:
   :inherited-members:


Recording data
==============

Probes
~~~~~~

.. autoclass:: navground_learning.probes.GymProbe
   :members:
   :inherited-members:

.. autoclass:: navground_learning.probes.RewardProbe
   :members:
   :inherited-members:

Experiments
~~~~~~~~~~~

.. autofunction:: navground_learning.evaluate.make_experiment

.. autofunction:: navground_learning.evaluate.make_experiment_with_env

.. autofunction:: navground_learning.evaluate.evaluate_expert


Rollouts
~~~~~~~~

.. autofunction:: navground_learning.rollout.get_trajectories_from_run

.. autofunction:: navground_learning.rollout.get_trajectories_from_experiment


Models
======

.. autoclass:: navground_learning.policies.OrderInvariantCombinedExtractor
   :members:
   :show-inheritance:


Scenarios
=========

CorridorWithObstacle
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: navground_learning.scenarios.CorridorWithObstacle
   :members:
   :show-inheritance:

Forward
~~~~~~~

.. autoclass:: navground_learning.scenarios.ForwardScenario
   :members:
   :show-inheritance:

Evaluation
~~~~~~~~~~

.. autoclass:: navground_learning.scenarios.EvaluationScenario
   :members:
   :show-inheritance:



