from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from ..types import AnyPolicyPredictor

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame


class LogField(NamedTuple):
    key: str
    label: str
    low: float | None = None
    high: float | None = None
    linestyle: str = '-'
    color: str | None = None


def plot_logs(logs: DataFrame,
              key: str,
              fields: Sequence[LogField],
              two_axis: bool = False,
              title: str = '',
              **kwargs: Any) -> None:
    """
    Plots logged fields.

    :param      logs:     The logs
    :param      key:      Common x-axis key
    :param      fields:   Which fields to plot.
    :param      two_axis: Whether to use two axis (only if there are two fields)

    """
    ts = logs[key]
    if two_axis and len(fields) != 2:
        print("two_axis=True only valid when plotting 2 fields!",
              file=sys.stderr)
        two_axis = False
    if two_axis:
        fig, ax = plt.subplots(**kwargs)
        for i, field in enumerate(fields):
            vs = logs[field.key]
            if i == 1:
                ax = ax.twinx()
            ax.plot(ts,
                    vs,
                    color=field.color,
                    linestyle=field.linestyle,
                    label=field.label)
            if i == 0:
                ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                          loc='lower left')
                ax.set_xlabel('time steps')
            else:
                ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                          loc='lower right')
            ax.set_ylabel(field.label)
        if title:
            plt.title(title)
    else:
        _ = plt.figure(**kwargs)
        scaled = False
        for field in fields:
            vs = logs[field.key]
            if field.low is not None and field.high is not None:
                vs = (vs - field.low) / (field.high - field.low)
                label = f'{field.label} (scaled)'
                scaled = True
            else:
                label = field.label
            plt.plot(ts,
                     vs,
                     color=field.color,
                     linestyle=field.linestyle,
                     label=label)
        if scaled:
            plt.ylim(-0.1, 1.1)
        plt.xlabel('time steps')
        if title:
            plt.title(title)
        plt.legend()


def plot_policy(policy: AnyPolicyPredictor,
                axs: Iterable[Axes] | None = None,
                fix: dict[str, float | Iterable[float]] = {},
                colors: Iterable | None = None,
                variable: dict[str, tuple[float, float]] = {},
                actions: dict[int, str] = {},
                cmap: str = 'RdYlGn',
                samples: int = 101,
                width: float = 5,
                height: float = 3,
                label: str = '',
                **kwargs: Any):
    in_space = policy.observation_space
    out_space = policy.action_space
    if not isinstance(in_space, gym.spaces.Dict):
        raise NotImplementedError
    variable_space = {k: v for k, v in in_space.items() if k not in fix}
    in_num = len(variable_space)
    if in_num > 2:
        raise ValueError(f"Too many variables: {', '.join(variable_space)}!")
    if in_num == 0:
        raise ValueError("No variable")
    for k, s in variable_space.items():
        space = cast('gym.spaces.Box', s)
        if k not in variable:
            variable[k] = (space.low.flatten()[0], space.high.flatten()[0])
    out_num = len(actions)
    for k, v in fix.items():
        if isinstance(v, Iterable):
            vs = list(v)
            if len(vs) == 0:
                raise ValueError(f"Empty sequence for fixed {k}")
            elif len(vs) == 1:
                fix[k] = vs[0]
                continue
            else:
                if in_num == 2:
                    if not axs:
                        figsize = (width * out_num, height * len(vs))
                        _, all_axs = plt.subplots(ncols=out_num,
                                                  nrows=len(vs),
                                                  figsize=figsize)
                        all_axs = np.atleast_2d(all_axs)
                    else:
                        all_axs = axs
                    for f_v, f_axs in zip(vs, all_axs.T, strict=True):
                        f_fix = dict(**fix)
                        f_fix[k] = f_v
                        plot_policy(policy=policy,
                                    axs=f_axs,
                                    fix=f_fix,
                                    variable=variable,
                                    actions=actions,
                                    cmap=cmap,
                                    samples=samples,
                                    label=f'{k}={round(f_v, 2)}')
                    return
                else:
                    if axs is None:
                        figsize = (width * out_num, height)
                        _, all_axs = plt.subplots(ncols=out_num,
                                                  nrows=1,
                                                  figsize=figsize)
                        all_axs = np.atleast_1d(all_axs)
                    else:
                        all_axs = axs
                    if colors is None:
                        colors = [None] * len(vs)
                    for f_v, color in zip(vs, colors, strict=True):
                        f_fix = dict(**fix)
                        f_fix[k] = f_v
                        plot_policy(policy,
                                    axs=all_axs,
                                    variable=variable,
                                    actions=actions,
                                    cmap=cmap,
                                    samples=samples,
                                    fix=f_fix,
                                    label=f'{k}={round(f_v, 2)}',
                                    color=color)
                    for ax in all_axs:
                        ax.legend()
                    return
    if axs is None:
        figsize = (width * out_num, height)
        _, all_axs = plt.subplots(ncols=out_num, nrows=1, figsize=figsize)
        axs = np.atleast_1d(all_axs)
    obs = {}
    variables = {
        k: np.linspace(low, high, samples, dtype=np.float32)
        for k, (low, high) in variable.items()
    }
    if in_num == 2:
        xs, ys = list(variables.values())
        obs = {
            k: v
            for k, v in zip(variables, np.meshgrid(xs, ys), strict=True)
        }
        size = samples**2
    else:
        obs = dict(**variables)
        size = samples
    for k, v in fix.items():
        obs[k] = np.full(size, v, dtype=np.float32)
    obs = {
        k: obs[k].reshape(-1,
                          *cast('gym.spaces.Box', v).shape)
        for k, v in in_space.items()
    }
    act, _ = policy.predict(obs, deterministic=True)
    if in_num == 2:
        act = act.reshape(samples, samples, -1)
    if isinstance(out_space, gym.spaces.Discrete):
        act = act - 1
    elif isinstance(out_space, gym.spaces.MultiBinary):
        shape = act.shape
        act = act.reshape(*shape[:-1], -1, 2)
        act = act[..., 1] - act[..., 0]
    if in_num == 2:
        (xlabel, xs), (ylabel, ys) = list(variables.items())

        for ax, (i, out_label) in zip(axs, actions.items(), strict=True):
            low = -1
            high = 1
            im = ax.imshow(act[::-1, :, i],
                           vmin=low,
                           vmax=high,
                           cmap=cmap,
                           **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax)
            ps = np.linspace(0, samples, 5)
            xs = np.linspace(xs[0], xs[-1], 5)
            ys = np.linspace(ys[0], ys[-1], 5)
            ax.set_xticks(ps, [f'{x:.2f}' for x in xs])
            ax.set_yticks(ps[::-1], [f'{y:.2f}' for y in ys])
            if label:
                title = f'{out_label} ({label})'
            else:
                title = out_label
            ax.set_title(title)
    else:
        xlabel, xs = next(iter(variables.items()))
        for ax, (i, out_label) in zip(axs, actions.items(), strict=True):
            ax.plot(xs, act[..., i], label=label, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(out_label)
            ax.set_title(out_label)
