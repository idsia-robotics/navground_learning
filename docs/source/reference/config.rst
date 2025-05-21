=============
Configuration
=============

:py:mod:`!navground.learning.config`

.. py:module:: navground.learning.config

Actions
=======

Base class
----------

.. autoclass:: ActionConfig
   :members:
   :exclude-members: __init__

End-to-end
----------

Continuous
""""""""""

.. autoclass:: ControlActionConfig
   :members:

Discrete
""""""""

.. autoclass:: DiscreteControlActionConfig
   :members:
   :show-inheritance:

[Multi-]Binary
""""""""""""""

.. autoclass:: BinaryControlActionConfig
   :members:
   :show-inheritance:

Communication
-------------

Continuous
""""""""""

.. autoclass:: ControlActionWithCommConfig
   :members:
   :show-inheritance:

Discrete
""""""""

.. autoclass:: DiscreteControlActionWithCommConfig
   :members:
   :show-inheritance:

Behavior Modulation
-------------------

.. autoclass:: ModulationActionConfig
   :members:

Observations
============

Base class
----------

.. autoclass:: ObservationConfig
   :members:

Default
-------

.. autoclass:: DefaultObservationConfig
   :members:

State
=====

Base class
----------

.. autoclass:: StateConfig
   :members:

Default
-------

.. autoclass:: DefaultStateConfig
   :members:

Group
=====

.. autoclass:: GroupConfig
   :members:

.. autofunction:: merge_groups_configs


