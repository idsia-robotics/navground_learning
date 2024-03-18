============
Installation
============

Install the package dependencies:

- ``navground_sim``: `instructions <https://idsia-robotics.github.io/navground/_build/html/installation.html#simulation-c-and-python>`
- ``gymnasium`` (required), ``imitation`` (optional) and ``stable-baseline3`` (optional)

  .. code-block:: console

  	 pip install gymnasium imitation stable-baselines3


Then install the package, for instance using colcon

.. code-block:: console

	colcon build --merge-install --packages-select navground_learning