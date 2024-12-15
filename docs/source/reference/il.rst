==================
Imitation Learning
==================

:py:mod:`!navground.learning.il`

.. py:module:: navground.learning.il

In this sub-package, we wraps and patch parts of `imitation <https://imitation.readthedocs.io>`_ for algorithms to

- expose a similar API has :py:class:`stable_baselines3.common.base_class.BaseAlgorithm`, in particular for saving and loading

- accept as experts callables that consumes an ``info`` dictionary, like in particular :py:class:`navground.learning.policies.info_predictor.InfoPolicy`, which is the way we expose actions computed in navground thought the environments, and therefore the way to learn to imitate navigation behaviors running in navground:

  .. py:type:: PolicyCallableWithInfo
     :canonical: typing.Callable[[Observation, State | None, EpisodeStart, Info | None], tuple[Action, State | None]]

  This extend expert to any policy in 

  .. py:type:: AnyPolicy
     :canonical: stable_baselines3.common.base_class.BaseAlgorithm | stable_baselines3.common.policies.BasePolicy | PolicyCallable | PolicyCallableWithInfo

  where

  .. py:type:: PolicyCallable
     :canonical: typing.Callable[[Observation, State | None, EpisodeStart], tuple[Action, State | None]]

  is the same as :py:type:`imitation.data.rollout.PolicyCallable`


This requires a modified version of `rollout.py <https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/data/rollout.py>`_, which we implement in function

.. autofunction:: navground.learning.il.rollout.generate_trajectories

.. note::

   The original functionality of ``imitation`` is maintained. Experts that do not accept an ``info`` dictionary are still accepted.

Utilities
=========

.. autofunction:: make_vec_from_env

.. autofunction:: make_vec_from_penv

.. autofunction:: setup_tqdm

Base class
===========

.. autoclass:: BaseILAlgorithm
   :members:

Behavior Cloning
================

.. autoclass:: BC
   :members:

DAgger
======

.. autoclass:: DAgger
   :members:
