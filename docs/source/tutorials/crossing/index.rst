========
Crossing
========

In these notebooks, we take a look at a more challenging scenario where we learn to navigate among *many* agents. For the same scenario

.. literalinclude:: scenario.yaml
   :language: yaml

and sensor

.. literalinclude:: sensor.yaml
   :language: yaml

we try different algorithms to learn a navigation policy. In particular, we make use of the :py:class:`parallel multi-agent environment <navground.learning.parallel_env.MultiAgentNavgroundEnv>` to make all agents in the group learn a policy *in parallel*.


.. toctree::
   :maxdepth: 2

   Training-SA
   Analysis-SA
   Training-MA
   Analysis-MA     
