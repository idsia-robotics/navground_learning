from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

    from ...env import BaseEnv
    from ...parallel_env import BaseParallelEnv
    from ...types import PathLike, GroupObservationsTransform, ObservationTransform

from ...evaluation import make_experiment_with_env, record_run_video


class VideoCallback(BaseCallback):
    """
    A SB3 callback that makes a video of one or more runs
    of an environment.
    """

    def __init__(self,
                 env: BaseEnv | BaseParallelEnv | VecEnv,
                 save_path: PathLike | None = None,
                 number: int = 1,
                 grouped: bool = False,
                 pre: ObservationTransform | None = None,
                 group_pre: GroupObservationsTransform | None = None,
                 video_format: str = 'mp4',
                 policy: Any = None,
                 **kwargs: Any) -> None:
        """
        Constructs a new instance.

        :param      env:           The environment
        :param      save_path:     Where to save the videos
        :param      number:        The number of videos per evaluation
        :param      grouped:       Whether the policy is grouped
        :param      pre:           An optional transformation to apply to observations
                                   of all individual agents
        :param      group_pre:     An optional transformation to apply to observations
                                   of all groups
        :param      video_format:  The video format
        :param      policy:        The policy
        :param      kwargs:        The keywords arguments passed to the renderer
        """
        super().__init__()
        self.exp = None
        self.save_path = None
        self.number = number
        self.env = env
        self.format = video_format
        self.render_kwargs = kwargs
        self.grouped = grouped
        self.policy = policy
        self.pre = pre
        self.group_pre = group_pre,

    def _on_step(self):
        if self.save_path is None:
            self.save_path = Path(self.model.logger.get_dir()) / 'videos'
            self.save_path.mkdir(parents=True, exist_ok=True)
        if self.exp is None:
            policy = self.policy or self.model.policy
            self.exp = make_experiment_with_env(self.env,
                                                policy=policy,
                                                record_reward=False,
                                                grouped=self.grouped,
                                                record_success=False,
                                                pre=self.pre,
                                                group_pre=self.group_pre)
        for i in range(self.number):
            name = f'{self.num_timesteps}_{i}.{self.format}'
            record_run_video(self.exp,
                             path=self.save_path / name,
                             seed=i,
                             **self.render_kwargs)
        return True
