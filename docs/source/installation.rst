============
Installation
============

Install the package dependencies:

- ``navground_sim``: `instructions <https://idsia-robotics.github.io/navground/_build/html/installation.html#simulation-c-and-python>`_
- ``gymnasium`` (required), ``imitation`` (required for imitation learning), ``stable-baseline3`` (optional, for reinforcement learning), ``pettingzoo`` (required for multi-agent envs), ``supersuit`` (required for multi-agent imitation learning)

  .. code-block:: console

  	 pip install gymnasium imitation stable-baselines3 pettingzoo SuperSuit


Then install the package, for instance using colcon

.. code-block:: console

	colcon build --merge-install --packages-select navground_learning