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
   :inherited-members:

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
   :inherited-members:

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
   :inherited-members:

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
   :inherited-members:

Group
=====

.. autoclass:: GroupConfig
   :members:
   :inherited-members:

.. autofunction:: merge_groups_configs


