=========
Reference
=========

Environment
===========

.. autoclass:: navground_learning.env.GymAgentConfig
   :members:

.. autoclass:: navground_learning.env.NavgroundEnv
   :members:
   :show-inheritance:

Reward functions
================

.. autofunction:: navground_learning.env.null_reward

.. autofunction:: navground_learning.env.social_reward

Inference
=========

.. autoclass:: navground_learning.behaviors.PolicyBehavior
   :members:
   :show-inheritance:

Training
========

Behavior cloning
----------------

.. autoclass:: navground_learning.il.bc.Trainer
   :members:
   :inherited-members:

DAgger
------

.. autoclass:: navground_learning.il.dagger.Trainer
   :members:
   :inherited-members: