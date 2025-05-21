from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from ...onnx import export


class ExportOnnxCallback(BaseCallback):
    """
    Exports the (best) model policy as "best_policy.onnx".
    """

    def _on_step(self) -> bool:
        p = self.model.logger.get_dir()
        if p is not None:
            path = Path(p) / 'best_policy.onnx'
            export(self.model.policy, path)
        return True
