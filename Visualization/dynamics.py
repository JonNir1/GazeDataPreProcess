import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import Utils.timeseries_utils as tsutils
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent
from GazeEvents.FixationEvent import FixationEvent


def velocity_profile(events: List[BaseVisualGazeEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    velocities = [e.get_velocity_series() for e in events]
    kwargs['ylabel'] = "Velocity (px/s)"
    kwargs['title'] = kwargs.get('title', "Velocity Dynamics")
    ax = dynamic_profile(velocities, ax, **kwargs)
    return ax


def acceleration_profile(events: List[BaseVisualGazeEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    # TODO: calculate accelerations and plot them
    kwargs['ylabel'] = "Acceleration (px/s^2)"
    kwargs['title'] = kwargs.get('title', "Acceleration Dynamics")
    raise NotImplementedError


def pupil_size_profile(fixations: List[FixationEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    pupil_sizes = [f.get_pupil_series() for f in fixations]
    kwargs['ylabel'] = "Pupil Size (mm)"
    kwargs['title'] = kwargs.get('title', "Pupil Size Dynamics")
    ax = dynamic_profile(pupil_sizes, ax, **kwargs)
    return ax


def dynamic_profile(timeseries: List[pd.Series], ax: plt.Axes, **kwargs) -> plt.Axes:
    timeseries_df = tsutils.interpolate_and_merge_timeseries(timeseries, interpolation_kind=kwargs.get('interpolation_kind', 'linear'))
    mean = timeseries_df.mean(axis=1)
    sem = timeseries_df.sem(axis=1)

    primary_color = kwargs.get('primary_color', '#034e7b')  # default: dark blue
    primary_linewidth = kwargs.get('primary_linewidth', 2)
    ax.plot(mean.index, mean, color=primary_color, linewidth=primary_linewidth, zorder=10, label='Mean')
    ax.fill_between(mean.index, mean - sem, mean + sem, color=primary_color, alpha=0.5, zorder=2)

    if kwargs.get('show_individual', False):
        secondary_color = kwargs.get('secondary_color', '#a6bddb')  # default: light blue
        secondary_linewidth = kwargs.get('secondary_linewidth', 1)
        for col in timeseries_df.columns:
            ax.plot(timeseries_df.index, timeseries_df[col], color=secondary_color,
                    linewidth=secondary_linewidth, alpha=0.75, zorder=1)

    label_size = kwargs.get('label_size', 12)
    tick_size = kwargs.get('tick_size', 10)
    ax.set_xlabel("Relative Time (%)", fontsize=label_size)
    ax.set_ylabel(kwargs.get('ylabel', ""), fontsize=label_size)
    ax.set_xticks(ticks=np.arange(0, 110, 10), labels=np.arange(0, 110, 10))
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(kwargs.get('title', ""), fontsize=kwargs.get('title_size', 14))
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(bottom=max([mean.min() - 2 * sem.max(), y_bottom]), top=min([mean.max() + 2 * sem.max(), y_top]))
    return ax

