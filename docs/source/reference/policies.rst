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
