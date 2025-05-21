=====
Utils
=====

:py:mod:`!navground.learning.utils`

.. py:module:: navground.learning.utils

.. 
  Jupyter
  =======
  
  :py:mod:`!navground.learning.utils.jupyter`
  
  .. py:module:: navground.learning.utils.jupyter
  
  .. autofunction:: skip_if
  
  .. autofunction:: run_if
  
  .. autofunction:: run_and_time_if
  
  .. autofunction:: clean_tqdm_rich

Plotting
========

:py:mod:`!navground.learning.utils.plot`

.. py:module:: navground.learning.utils.plot

.. autofunction:: plot_policy

.. autoclass:: LogField

.. autofunction:: plot_logs


StableBaseLine3
===============

:py:mod:`!navground.learning.utils.sb3`

.. py:module:: navground.learning.utils.sb3

.. autoclass:: ExportOnnxCallback

.. autoclass:: ProgressBarWithRewardCallback

.. autoclass:: VideoCallback

.. autofunction:: load_eval_logs

.. autofunction:: plot_eval_logs

BenchMARL
=========

:py:mod:`!navground.learning.utils.benchmarl`

.. py:module:: navground.learning.utils.benchmarl

.. autoclass:: NavgroundExperiment
   :members:

.. autoclass:: SingleAgentPolicy
   :members:

.. autoclass:: ExportPolicyCallback

.. autoclass:: AlternateActorCallback

.. autofunction:: make_env

.. autofunction:: evaluate_policy

