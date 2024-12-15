============
Installation
============

Using pip
=========

The package is released on PyPi

.. code-block:: console

	 pip install navground_learning[all]

If you prefer to install the latest developments, install it from github

.. code-block:: console

	 pip install git+https://github.com/idsia-robotics/navground_learning.git@main

Using colcon
============

Install the package dependencies:

- ``navground_sim``: `instructions <https://idsia-robotics.github.io/navground/_build/html/installation.html#simulation-c-and-python>`_

- ``gymnasium`` and ``pettingzoo`` (required), ``imitation`` (for imitation learning), ``stable-baseline3`` (for reinforcement learning),  ``supersuit`` (for multi-agent reinforcement/imitation learning)

  .. code-block:: console

  	 pip install gymnasium imitation stable-baselines3 pettingzoo SuperSuit onnxruntime

Then install this package

.. code-block:: console

	 colcon build --merge-install --packages-select navground_learning