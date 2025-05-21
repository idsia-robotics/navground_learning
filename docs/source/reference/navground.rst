====================
Navground Components
====================

Behaviors
=========

:py:mod:`!navground.learning.behaviors`

.. py:module:: navground.learning.behaviors

PolicyBehavior
~~~~~~~~~~~~~~

.. autoclass:: PolicyBehavior
   :members:
   :show-inheritance:

GroupedPolicyBehavior
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GroupedPolicyBehavior
   :members:
   :show-inheritance:

Pad
~~~

:py:mod:`!navground.learning.behaviors.pad`

.. py:module:: navground.learning.behaviors.pad

.. autoclass:: DistributedPadBehavior
   :members:
   :show-inheritance:

.. autoclass:: CentralizedPadBehavior
   :members:
   :show-inheritance:

.. autoclass:: StopAtPadBehavior
   :members:
   :show-inheritance:

Scenarios
=========

:py:mod:`!navground.learning.scenarios`

.. py:module:: navground.learning.scenarios

CorridorWithObstacle
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CorridorWithObstacle
   :members:
   :show-inheritance:

Forward
~~~~~~~

.. autoclass:: ForwardScenario
   :members:
   :show-inheritance:

Pad
~~~

.. autoclass:: PadScenario
   :members:
   :show-inheritance:

.. autofunction:: navground.learning.scenarios.pad.render_kwargs


State Estimations
=================

:py:mod:`!navground.learning.state_estimations`

.. py:module:: navground.learning.state_estimations


.. autoclass:: CommSensor
   :members:

Probes
======

:py:mod:`!navground.learning.probes`

.. py:module:: navground.learning.probes


.. autoclass:: GymProbe
   :members:

.. autoclass:: RewardProbe
   :members:

.. autoclass:: SuccessProbe
   :members:

.. autoclass:: CommProbe
   :members:
