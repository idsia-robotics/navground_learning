===========================
Exclusive crossing on a pad
===========================

In this longer series of notebook, we look at a scenario where two agents must coordinate to not step at the same time on a pad while crossing each other along a corridor. The motion is restricted to one dimension and the agents always know the relative position of the pad.

We consider several cases defined by which information the agents known about each other and which type of action they can take, including cases where they can (learn to) explicitly share information.

These notebook serve two goals:

1. showcasing different variants and algorithms that can be used with ``navground-learning``;

2. study minimal yet interesting learning tasks, in particular those with explicit communication which are difficult even for SoA multi-agent RL algorithms.

Policies
========

We will use the following notation to represent policies:

- 🟧 = observations by (orange) agent  
- 🔶 = actions by (orange) agent 
- 🟠 = message broadcasted by (orange) agent

We are going to look at the following variants:

- Distributed policy

  🟧 -> 🔶

- Centralized policy

  🟧 + 🟦 -> 🔶 + 🔷


- Distributed policy with communication:

  🟧 + 🔵 -> 🔶 + 🟠 

- Distributed policy with unidirectional communication:

  The agents use different policies. One agent speaks (orange), the other listen (blue)

  O:      🟧  -> 🔶 + 🟠 

  B: 🟦 + 🟠  -> 🔷


.. toctree::
   :maxdepth: 2

   Scenario
   Behaviors
   SingleAgent/index
   Centralized/Centralized
   Distributed/index
   Communication/index
