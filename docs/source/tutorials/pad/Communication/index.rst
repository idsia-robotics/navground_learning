=====================================
Distributed policy with communication
=====================================

In the following notebooks about *communication*, the agents do not perceive each other directly but can exchange either a single float or a single bit. 

🟧 + 🔵 -> 🔶 + 🟠 

Implicitly, they learn to

- encode their state in a message that is usefull for the other agent
- decode the recevied message and to navigate safely (i.e., avoiding stepping on the pad at the same time).

To promote exchanging usefull information, during training, we can add a bit of the (efficacy) neighbor penalty to the agent reward.

Instability
===========

Learning to listen while learning to speak causes further instability to a learning task
that is already instable because of the presence of evolving agents.

We will look at two strategies reduce instability: centralized training and model splitting


Centralized training
--------------------

We will test training modalities:

- learn the distributed policy (as we have done previously) in the same environment
- learn the distributed policy in central environment, casting it as a single agent task. 

  In this case, the policy has two separated modules A (action net) and C (comm net),
    that are evaluated in sequence like

    C:      🟧 -> 🟠  

    A: 🟧 + 🔵 -> 🔶

    During training, they are evaluated together for both agents, like they would form a centralized policy: 🟧 + 🟦 -> 🔶 + 🔷

    During inference, they are instead evaluated by each agent separately, which then exchange the message explicitly: 🟧 + 🔵 (from previous step) -> 🔶 (from A) + 🟠 (from C)


Split Models
------------

We will look at different NN models

- Action and message computed by a single MLP
- Action and message computed by a separated MLP, where the message does not use the receiving messsage as input. 

  *Splitting* the two models is usefull to stabilize the training but allow also to specify the model and the learning with more granularity, such as using different learning rate or different model complexity for the two outputs.


.. toctree::
   :maxdepth: 2

   parallel_sac
   Comm-PPO-Discrete
   Comm-SAC-CentralizedTraining
   benchmarl