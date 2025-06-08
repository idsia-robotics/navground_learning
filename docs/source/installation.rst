============
Installation
============

We support Python>=3.10. Users should prefer Python<=3.12 because important third-party packages like Stable-BaseLine3 or PyTorch do not support Python3.13 yet.

The package is released on PyPi

.. code-block:: console

	 pip install navground_learning[all]

If you prefer to install the latest developments from github:

.. code-block:: console

	 pip install git+https://github.com/idsia-robotics/navground_learning.git@main[all]


Install our fork of BenchMARL that adds:

- possible testing environments that are different that the training environments
- logging success rates

.. code-block:: console

   pip install git+https://github.com/jeguzzi/BenchMARL.git@main