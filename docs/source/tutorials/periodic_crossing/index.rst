=================
Periodic Crossing
=================

We look at another type of crossing scenario, where the agents move in a periodic world

.. literalinclude:: scenario.yaml
   :language: yaml


like a squared periodic "plaza" where agents coming from four different streams cross.

We use the same sensor as in the :doc:`../crossing/index` tutorial.

.. literalinclude:: sensor.yaml
   :language: yaml

In two notebooks, we train a policy using Reinforcement Learning 
to navigate this scenario: at first when all agents share the same target speed, and then when they have individual target speeds.


.. toctree::
   :maxdepth: 2

   SameSpeed 
   DifferentSpeed     
