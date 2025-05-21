========
Policies
========

:py:mod:`!navground.learning.policies`

.. py:module:: navground.learning.policies

Null
====

.. autoclass:: navground.learning.policies.null_policy.NullPolicy
   :members:
   :show-inheritance:

.. autoclass:: navground.learning.policies.null_predictor.NullPredictor
   :members:

Random
======

.. autoclass:: navground.learning.policies.random_policy.RandomPolicy
   :members:
   :show-inheritance:

.. autoclass:: navground.learning.policies.random_predictor.RandomPredictor
   :members:

Info
====

.. autoclass:: navground.learning.policies.info_predictor.InfoPolicy
   :members:

Ordering-invariant extractor
============================

.. py:type:: Reduction
   :canonical: typing.Callable[[torch.Tensor, int, bool], torch.Tensor]


.. autoclass:: navground.learning.policies.order_invariant.OrderInvariantCombinedExtractor
   :members:

.. autoclass:: navground.learning.policies.order_invariant.OrderInvariantFlattenExtractor
   :members:

.. autofunction:: navground.learning.policies.order_invariant.make_order_invariant_flatten_extractor


Centralized training with communication (SAC)
=============================================

.. autoclass:: navground.learning.policies.centralized_policy_with_comm.SACPolicyWithComm
   :members:

.. autoclass:: navground.learning.policies.centralized_policy_with_comm.DistributedCommPolicy
   :members:

Split MLP Policy (SAC)
======================

.. py:type:: InputSpec
   :module: navground.learning.policies.split_sac_policy
   :canonical: slice | Collection[str] | None

   Which inputs to use: a slice of a box observation space, a collection of keys of a dict observation or all (for None).

.. py:type:: NetArch
   :module: navground.learning.policies.split_sac_policy
   :canonical: list[int] | dict[str, list[int]] | None

   An (optional) network architecture


.. py:type:: ActorSpec
   :module: navground.learning.policies.split_sac_policy
   :canonical: tuple[int, InputSpec, NetArch]

   The specifics of a sub-module: output size, input specs and network architecture.

.. autoclass:: navground.learning.policies.split_sac_policy.SplitSACPolicy
   :members:

.. autoclass:: navground.learning.policies.split_sac_policy.AlternateActorCallback
   :members:

