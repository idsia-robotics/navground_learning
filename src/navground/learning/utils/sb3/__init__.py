from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stable_baselines3.common.callbacks import EvalCallback

from ...types import PathLike
from .export_onnx_callback import ExportOnnxCallback
from .log import load_eval_logs, plot_eval_logs, plot_rollout_logs
from .progress_bar_with_reward import ProgressBarWithRewardCallback
from .video_callback import VideoCallback

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv


def callbacks(venv: VecEnv,
              best_model_save_path: PathLike,
              eval_freq: int = 1000,
              n_eval_episodes: int = 30,
              number_of_videos: int = 1,
              update_bar_every: int = 1,
              video_policy: Any = None,
              video_env: Any = None,
              export_to_onnx: bool = False,
              **kwargs: Any) -> list[BaseCallback]:
    if number_of_videos > 0:
        video_cb = VideoCallback(video_env or venv,
                                 number=number_of_videos,
                                 factor=4,
                                 policy=video_policy,
                                 **kwargs)
    else:
        video_cb = None
    callback_on_new_best = ExportOnnxCallback() if export_to_onnx else None
    eval_cb = EvalCallback(venv,
                           best_model_save_path=str(best_model_save_path),
                           eval_freq=eval_freq,
                           deterministic=True,
                           render=False,
                           n_eval_episodes=n_eval_episodes,
                           verbose=0,
                           callback_after_eval=video_cb,
                           callback_on_new_best=callback_on_new_best)
    return [eval_cb, ProgressBarWithRewardCallback(every=update_bar_every)]


__all__ = [
    "ProgressBarWithRewardCallback", "ExportOnnxCallback", "load_eval_logs",
    "plot_eval_logs", "VideoCallback", 'callbacks', 'plot_rollout_logs'
]
