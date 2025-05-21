from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ...types import PathLike
from ..plot import LogField, plot_logs


def load_logs(path: PathLike, key: str) -> pd.DataFrame:
    """
    Loads logs from a csv.

    :param      path:  The directory with csv logs
    :param      key:   The prefix of the fields to load

    :returns:   A dataframe with the logs
    """
    progress = pd.read_csv(Path(path) / 'progress.csv')
    cols = [k for k in progress.columns if key in k]
    keys = ['time/total_timesteps'] + cols
    fs = ~progress[cols[0]].isna()
    f_progress = progress[keys].loc[fs]
    return f_progress.fillna(0)


def load_eval_logs(path: PathLike) -> pd.DataFrame:
    """
    Loads the evaluation logs from a csv.

    :param      path:  The directory with csv logs

    :returns:   A dataframe with the logs
    """
    return load_logs(path, key='eval')


def load_rollout_logs(path: PathLike) -> pd.DataFrame:
    """
    Loads the rollout logs from a csv.

    :param      path:  The directory with csv logs

    :returns:   A dataframe with the logs
    """
    return load_logs(path, key='rollout')


def plot_eval_logs(path: PathLike,
                   two_axis: bool = False,
                   reward: bool = True,
                   reward_color: str = 'k',
                   reward_linestyle: str = '--',
                   success: bool = False,
                   length: bool = False,
                   reward_low: float | None = None,
                   reward_high: float = 0,
                   length_low: float = 0,
                   length_high: float | None = None,
                   **kwargs: Any) -> None:
    """
    Plots reward, success and/or length from the eval logs

    :param      path:         The directory with csv logs
    :param      reward:       Whether to plot the reward
    :param      reward_color: The reward color
    :param      reward_linestyle: The reward linestyle
    :param      success:      Whether to plot the success
    :param      length:       Whether to plot the length
    :param      reward_low:   An optional lower bound to scale the reward
    :param      reward_high:  An optional upper bound to scale the reward
    :param      lenght_low:   An optional lower bound to scale the reward
    :param      lenght_high:  An optional upper bound to scale the reward
    :param      two_axis:     Whether to use two axis (only if there are two fields)
    """
    fields: list[LogField] = []
    if reward:
        fields.append(
            LogField(label="mean reward",
                     key="eval/mean_reward",
                     linestyle=reward_linestyle,
                     color=reward_color,
                     low=reward_low,
                     high=reward_high))
    if success:
        fields.append(
            LogField(label="success rate",
                     key="eval/success_rate",
                     linestyle='-',
                     color='g'))
    if length:
        fields.append(
            LogField(label="mean length",
                     key="eval/mean_ep_length",
                     linestyle=':',
                     color='b',
                     low=length_low,
                     high=length_high))
    plot_logs(load_logs(path, key='eval'),
              title="eval",
              two_axis=two_axis,
              fields=fields,
              key='time/total_timesteps',
              **kwargs)


def plot_rollout_logs(path: PathLike,
                      two_axis: bool = False,
                      reward: bool = True,
                      success: bool = False,
                      length: bool = False,
                      reward_low: float | None = None,
                      reward_high: float = 0,
                      length_low: float = 0,
                      length_high: float | None = None,
                      **kwargs: Any) -> None:
    """
    Plots reward, success and/or length from the rollout logs

    :param      path:         The directory with csv logs
    :param      reward:       Whether to plot the reward
    :param      success:      Whether to plot the success
    :param      length:       Whether to plot the length
    :param      reward_low:   An optional lower bound to scale the reward
    :param      reward_high:  An optional upper bound to scale the reward
    :param      lenght_low:   An optional lower bound to scale the reward
    :param      lenght_high:  An optional upper bound to scale the reward
    :param      two_axis:     Whether to use two axis (only if there are two fields)
    """
    fields: list[LogField] = []
    if reward:
        fields.append(
            LogField(label="mean reward",
                     key="rollout/ep_rew_mean",
                     linestyle='--',
                     color='k',
                     low=reward_low,
                     high=reward_high))
    if success:
        fields.append(
            LogField(label="success rate",
                     key="rollout/success_rate",
                     linestyle='-',
                     color='g'))
    if length:
        fields.append(
            LogField(label="mean length",
                     key="rollout/ep_len_mean",
                     linestyle=':',
                     color='b',
                     low=length_low,
                     high=length_high))
    plot_logs(load_logs(path, key='rollout'),
              title="rollout",
              two_axis=two_axis,
              fields=fields,
              key='time/total_timesteps',
              **kwargs)
