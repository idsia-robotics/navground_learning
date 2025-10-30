from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from tqdm.auto import tqdm


class ProgressBarWithRewardCallback(BaseCallback):
    """
    Similar to SB3's own
    :py:class:`stable_baselines3.common.callbacks.ProgressBarCallback`, it
    displays a progress bar when training SB3 agent
    using tqdm but includes episodes mean reward and length.
    """
    pbar: tqdm[Any]

    def __init__(self, every: int = 1) -> None:
        super().__init__()
        self.every = every

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] -
                         self.model.num_timesteps)
        self.pbar.mininterval = 1
        self._steps = 0
        self._count = 0

    def _on_step(self) -> bool:
        self._steps += self.training_env.num_envs
        # Update progress bar, we do num_envs steps per call to `env.step()`
        # self.pbar.update(self.training_env.num_envs)
        return True

    def _on_rollout_end(self) -> None:
        if self._count % self.every:
            self._count += 1
            return
        if self.model.ep_info_buffer:
            rs = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            ls = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
            mr = safe_mean(rs)
            sr = 0 if len(rs) == 0 else float(np.std(rs))
            ml = safe_mean(ls)
            sl = 0 if len(ls) == 0 else int(np.std(ls))
            if self.model.ep_success_buffer:
                s = f'({safe_mean(self.model.ep_success_buffer):.2})'
            else:
                s = ''
            self.pbar.set_description(
                f"Reward {mr:.1f} ± {sr:.1f}, Steps {ml:.0f} ± {sl} {s}")
        self.pbar.update(self._steps)
        self._steps = 0
        self._count += 1

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()
