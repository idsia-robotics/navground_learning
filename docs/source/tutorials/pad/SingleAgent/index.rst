===================================
Single ML agent meets `Dummy` agent
===================================

The yellow agent learns a policy, while the cyan agents applies the `Dummy` behavior instead, which just makes it move at full speed forward.

The policy computes (linear) accelerations. Because `Dummy` will always move at the same constant speed, the policy does not need to know the its speed.


    🟧 -> 🔶


.. toctree::
   :maxdepth: 2

   Dummy-Continuos
   Dummy-Discrete