from __future__ import annotations

from sphinx.addnodes import pending_xref

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'navground_learning'
copyright = '2024, Jerome Guzzi et al. (IDSIA, USI-SUPSI)'
author = 'Jerome Guzzi et al. (IDSIA, USI-SUPSI)'
release = '0.3.0.dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_copy_source = False

extensions = [
    'sphinx.ext.autodoc', 'sphinx_rtd_theme', 'nbsphinx',
    'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
exclude_patterns = [
    'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'tutorials/archive',
    'tutorials/antipodal', 'tutorials/corridor',
]

# autodoc_typehints_format = 'short'
# autodoc_member_order = 'groupwise'
# autodoc_class_signature = 'mixed'
add_module_names = False
autodoc_inherit_docstrings = False
autodoc_member_order = 'groupwise'
autodoc_class_signature = 'mixed'
autoclass_content = 'class'
autodoc_docstring_signature = True
autodoc_typehints = "both"
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = False
autodoc_default_options = {'show-inheritance': False}

autodoc_type_aliases = {
    'Array': 'Array',
    'Action': 'Action',
    'Observation': 'Observation',
    'State': 'State',
    'EpisodeStart': 'EpisodeStart',
    'Info': 'Info',
    'PathLike': 'PathLike',
    'AnyPolicyPredictor': 'AnyPolicyPredictor',
    'T': 'T',
    'Reduction': 'Reduction',
    'BaseEnv': 'BaseEnv',
    'BaseParallelEnv': 'BaseParallelEnv',
    'ObservationTransform': 'ObservationTransform'
}

intersphinx_mapping = {
    'gymnasium': ('https://gymnasium.farama.org', None),
    'navground': ('https://idsia-robotics.github.io/navground/latest', None),
    'pettingzoo': ('https://pettingzoo.farama.org', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://docs.pytorch.org/docs/stable', None),
    'imitation': ('https://imitation.readthedocs.io/en/latest/', None),
    'stable_baselines3':
    ('https://stable-baselines3.readthedocs.io/en/master/', None),
    'onnxruntime': ('https://onnxruntime.ai/docs/api/python', None),
    'benchmarl': ('https://benchmarl.readthedocs.io/en/latest', None),
    'torchrl': ('https://docs.pytorch.org/rl/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable', None)
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

reftarget_aliases = {}
reftarget_aliases['py'] = {
    'AlgorithmConfig': 'benchmarl.algorithms.common.AlgorithmConfig',
    'ExperimentConfig': 'benchmarl.experiment.ExperimentConfig',
    'ModelConfig': 'benchmarl.models.common.ModelConfig',
    'EnvBase': 'torchrl.envs.EnvBase',
    'gym.Env': 'gymnasium.Env',
    'Axes': 'matplotlib.axes.Axes',
    'Path': 'pathlib.Path',
    'pd.DataFrame': 'pandas.DataFrame',
    'DataFrame': 'pandas.DataFrame',
    'InputSpec': 'navground.learning.policies.split_sac_policy.InputSpec',
    'NetArch': 'navground.learning.policies.split_sac_policy.NetArch',
    'ActorSpec': 'navground.learning.policies.split_sac_policy.ActorSpec',
    'PyTorchObs': 'navground.learning.types.PyTorchObs',
    'Array': 'navground.learning.types.Array',
    'Action': 'navground.learning.types.Action',
    'Observation': 'navground.learning.types.Observation',
    'Info': 'navground.learning.types.Info',
    'EpisodeStart': 'navground.learning.types.EpisodeStart',
    'State': 'navground.learning.types.State',
    'ObservationTransform': 'navground.learning.types.ObservationTransform',
    'PathLike': 'navground.learning.types.PathLike',
    'IndicesLike': 'navground.learning.indices.IndicesLike',
    'Indices': 'navground.learning.indices.Indices',
    'AnyPolicy': 'navground.learning.il.AnyPolicy',
    'rollout.AnyPolicy': 'navground.learning.il.AnyPolicy',
    'PolicyCallable': 'navground.learning.il.PolicyCallable',
    'PolicyCallableWithInfo': 'navground.learning.il.PolicyCallableWithInfo',
    'Pose2.integrate': 'navground.core.Pose2.integrate',
    'Behavior': 'navground.core.Behavior',
    'compute_cmd_internal': 'navground.core.Behavior.compute_cmd_internal',
    'BehaviorModulation.pre': 'navground.core.BehaviorModulation.pre',
    'BehaviorModulation.post': 'navground.core.BehaviorModulation.post',
    'Kinematics.is_wheeled': 'navground.core.Kinematics.is_wheeled',
    'navground.learning.ObservationConfig':
    'navground.learning.config.ObservationConfig',
    'navground.learning.ActionConfig':
    'navground.learning.config.ActionConfig',
    'navground.learning.GroupConfig': 'navground.learning.config.GroupConfig',
    'navground.learning.ControlActionConfig':
    'navground.learning.config.ControlActionConfig',
    'navground.learning.ModulationActionConfig':
    'navground.learning.config.ModulationActionConfig',
    'PolicyPredictor': 'navground.learning.types.PolicyPredictor',
    'PolicyPredictorWithInfo':
    'navground.learning.types.PolicyPredictorWithInfo',
    'AnyPolicyPredictor': 'navground.learning.types.AnyPolicyPredictor',
    'sim.Sensor': 'navground.sim.Sensor',
    'sim.World': 'navground.sim.World',
    'gym.Space': 'gymnasium.spaces.Space',
    'gym.spaces.Dict': 'gymnasium.spaces.Dict',
    'gym.spaces.Box': 'gymnasium.spaces.Box',
    'dc.InitVar': 'dataclasses.InitVar',
    'nn.Module': 'torch.nn.Module',
    'th.Tensor': 'torch.Tensor',
    'VecEnv': 'stable_baselines3.common.vec_env.VecEnv',
    'sim.Experiment': 'navground.sim.Experiment',
    'sim.Scenario': 'navground.sim.Scenario',
    'sim.ExperimentalRun': 'navground.sim.ExperimentalRun',
    'sim.Agent': 'navground.sim.Agent',
    'np.random.Generator': 'numpy.random.Generator',
    'ParallelEnv': 'pettingzoo.utils.env.ParallelEnv',
    'pl.Path': 'pathlib.Path',
    'HierarchicalLogger': 'imitation.util.logger.HierarchicalLogger',
    'core.SensingState': 'navground.core.SensingState',
    'np.float64': 'numpy.float64',
    'np.float32': 'numpy.float32',
    "np.typing.NDArray": "numpy.typing.NDArray",
    'BaseParallelEnv': 'navground.learning.parallel_env.BaseParallelEnv',
    'BaselEnv': 'navground.learning.env.BaselEnv',
    'core.Kinematics': 'navground.core.Kinematics',
    'sim.Dataset': 'navground.sim.Dataset',
    'core.Behavior': 'navground.core.Behavior',
    'OffPolicyAlgorithm':
    'stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm',
    "types.TrajectoryWithRew": "imitation.data.types.TrajectoryWithRew",
    'NavgroundBaseEnv':
    'navground.learning.internal.base_env.NavgroundBaseEnv',
    'gymnasium.core.ActType': 'navground.learning.types.Action',
    'gymnasium.core.ObsType': 'navground.learning.types.Observation',
    'BaseAlgorithm': 'stable_baselines3.common.base_class.BaseAlgorithm',
    'th.optim.Optimizer': 'torch.optim.Optimizer',
}

_types = [
    'PathLike', 'AnyPolicyPredictor', 'Array', 'Action', 'Observation', 'Info',
    'EpisodeStart', 'State', "IndicesLike", "PolicyCallableWithInfo",
    'T', 'navground.learning.indices.T', 'BaseParallelEnv', 'BaseEnv',
    'Reduction', 'AnyPolicy', 'PolicyCallable', 'gymnasium.core.ObsType',
    'gymnasium.core.ActType', 'rollout.AnyPolicy', 'SensorLike',
    'SensorSequenceLike', 'TerminationCondition', 'ObservationTransform', 'PyTorchObs',
    'InputSpec', 'NetArch', 'ActorSpec'
]
_attrs = [
    'numpy.float64', 'numpy.float32', 'np.float32', 'np.float64', 'numpy.int8'
]
_data = ["numpy.typing.NDArray", "np.typing.NDArray"]


def resolve_internal_aliases(app, doctree):
    pending_xrefs = doctree.traverse(condition=pending_xref)
    for node in pending_xrefs:
        if node['refdomain'] == "py":
            if node['reftarget'] in _types:
                node["reftype"] = "type"
            elif node['reftarget'] in _attrs:
                node["reftype"] = "attr"
            elif node['reftarget'] in _data:
                node["reftype"] = "data"
    for node in pending_xrefs:
        alias = node.get('reftarget', None)
        d = node.get('refdomain', '')
        rs = reftarget_aliases.get(d, {})
        if alias is not None and alias in rs:
            node['reftarget'] = rs[alias]


_refs = {
    "BasePolicy":
    'https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html',
    "stable_baselines3.common.policies.BasePolicy":
    'https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html',
    "stable_baselines3.common.save_util.load_from_zip_file":
    "https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/save_util.py",
    "stable_baselines3.common.save_util.save_to_zip_file":
    "https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/save_util.py",
    "stable_baselines3.common.torch_layers.CombinedExtractor":
    "https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py",
    "stable_baselines3.common.type_aliases.PolicyPredictor":
    "https://github.com/DLR-RM/stable-baselines3/blob/9caa168686342ffc358c4acc7fbd842fc5fc8aac/stable_baselines3/common/type_aliases.py",
    "BaseFeaturesExtractor":
    "https://github.com/DLR-RM/stable-baselines3/blob/6af0601dc3c91d5e5477f10ca6976e77a24cccc8/stable_baselines3/common/torch_layers.py",
    "stable_baselines3.common.torch_layers.BaseFeaturesExtractor":
    "https://github.com/DLR-RM/stable-baselines3/blob/6af0601dc3c91d5e5477f10ca6976e77a24cccc8/stable_baselines3/common/torch_layers.py",
    "stable_baselines3.sac.policies.SACPolicy":
    "https://github.com/DLR-RM/stable-baselines3/blob/6af0601dc3c91d5e5477f10ca6976e77a24cccc8/stable_baselines3/sac/policies.py#L181",
    "Schedule":
    "https://github.com/DLR-RM/stable-baselines3/blob/6af0601dc3c91d5e5477f10ca6976e77a24cccc8/stable_baselines3/common/type_aliases.py#",
    "imitation.data.rollout.PolicyCallable":
    "https://imitation.readthedocs.io/en/latest/_modules/imitation/data/rollout.html",
    "rollout_without_info.GenTrajTerminationFn":
    "https://imitation.readthedocs.io/en/latest/_modules/imitation/data/rollout.html",
    "supersuit.vector.MarkovVectorEnv":
    "https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/markov_vector_wrapper.py",
    "supersuit.concat_vec_envs_v1":
    "https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/vector_constructors.py",
    "supersuit.pettingzoo_env_to_vec_env_v1":
    "https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/vector_constructors.py",
}


def missing_reference(app, env, node, contnode):
    from docutils import nodes
    t = node['reftarget']
    if t in _refs:
        refnode = nodes.reference('', '', internal=False, refuri=_refs[t])
        refnode.append(contnode)
        return refnode


def setup(app):
    app.connect('missing-reference', missing_reference)
    app.connect('doctree-read', resolve_internal_aliases)
