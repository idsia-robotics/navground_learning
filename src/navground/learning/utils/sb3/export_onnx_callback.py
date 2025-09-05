from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from ...onnx import export
from collections.abc import Callable


class ExportOnnxCallback(BaseCallback):
    """
    Exports the (best) model policy as "best_policy.onnx".
    """

    def __init__(self, name_fn: Callable[[BaseCallback], str]) -> None:
        super().__init__()
        self.name_fn = name_fn

    def _on_step(self) -> bool:
        p = self.model.logger.get_dir()
        if p is not None:
            path = Path(p) / (self.name_fn(self) + ".onnx")
            export(self.model.policy, path)
        return True
