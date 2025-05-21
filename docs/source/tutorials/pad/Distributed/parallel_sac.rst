============
Parallel SAC
============

We train a distributed policy, shared between the two agents, using SAC from StableBaseline3, in "parallel" mode, i.e., we treat the multi-agent evironement as a vectorized single agent environment performing rollouts in parallel, like we did for the :doc:`../../crossing/index` tutorial.

As for the previous notebooks, the policy computes (linear) accelerations. We test different observations spaces, action spaces and training algorithms. 


.. toctree::
   :maxdepth: 2

   Distributed-SAC
   Distributed-Blind-SAC
   Distributed-Position-SAC
   Distributed-Speed-SAC