=============
How to extend
=============

There are two main ways to extend the functionality and environments of `navground.learning`. 

Navground Components
====================

On the navground side, we can add components such as 

- new behaviors (to imitate)
- new sensors to include in observations
- new scenarios

In particular, each combination of sensor and scenario define a different environment to train policies.

The navground documentation `provides guides <https://idsia-robotics.github.io/navground/guides/extend/index.html>`_ on how to add components.

Configurations
==============

On the Gymnasium/PettingZoo side, we can subclass

- :py:class:`.ActionConfig` to translate actions to commands

- :py:class:`.ObservationConfig` to extract observations from sensing readings and behavior internal states

- :py:class:`.Reward` to add specific reward functions

Each of these component can be registered by name so to save/load environments to/from YAML.



