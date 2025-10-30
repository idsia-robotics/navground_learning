from __future__ import annotations

from torchrl.envs.libs.pettingzoo import PettingZooWrapper  # type: ignore

from benchmarl.experiment import Experiment  # type: ignore

from ...parallel_env import MultiAgentNavgroundEnv
from ...types import PathLike
from ...wrappers.name_wrapper import NameWrapper
from .callbacks import AlternateActorCallback, ExportPolicyCallback
from .evaluate import evaluate_policy
from .navground_experiment import NavgroundExperiment
from .policy import SingleAgentPolicy
from .split_mlp import SplitMlpConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchrl.envs import EnvBase  # type: ignore

def make_env(env: MultiAgentNavgroundEnv,
             seed: int = 0,
             categorical_actions: bool = False) -> EnvBase:
    return PettingZooWrapper(NameWrapper(MultiAgentNavgroundEnv(**env._spec)),
                             categorical_actions=categorical_actions,
                             device='cpu',
                             seed=seed,
                             return_state=env.has_state)


def reload_experiment(path: PathLike) -> Experiment:
    return Experiment.reload_from_file(str(path))


__all__ = [
    'ExportPolicyCallback', 'AlternateActorCallback', 'NavgroundExperiment',
    'SingleAgentPolicy', 'make_env', 'evaluate_policy', 'reload_experiment',
    'SplitMlpConfig'
]
