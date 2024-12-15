======================
Corridor with obstacle
======================

In a serie of two notebooks, we look at a simple but more interesting scenario than :doc:`../empty/empty`.

In the first notebook, we explore the scenario defined as

.. literalinclude:: scenario.yaml
   :language: yaml


where one agent wants to travel along a corridor with constant speed, avoiding the single obtacle.
We also get familiar with the inputs/outputs spaces of the models, which we train in the second notebook, 
to navigate using a combination of two sensors

.. literalinclude:: sensor.yaml
   :language: yaml

to detect the obstacle and the corridor walls.

.. toctree::
   :maxdepth: 2

   Scenario
   Learning
