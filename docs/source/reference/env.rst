==================================
Single-agent Gymnasium Environment
==================================

.. Base class
.. ==========

.. .. py:class:: NavgroundBaseEnv
..    :module: navground.learning.internal.base_env

..    Common internal base class for :py:class:`.NavgroundEnv` and
..    :py:class:`.MultiAgentNavgroundEnv` that integrate navground and gymnasium. 

.. Gymnasium Environment
.. =====================


:py:mod:`!navground.learning.env`

.. py:module:: navground.learning.env

.. py:type:: BaseEnv
   :canonical: gymnasium.Env[Observation, Action]

   The environment base class


.. autoclass:: NavgroundEnv
   :members:
   :inherited-members:
   :exclude-members: get_policy, get_wrapper_attr, has_wrapper_attr, set_wrapper_attr, np_random, np_random_seed, unwrapped
   
   .. :show-inheritance:
