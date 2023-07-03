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
    # TODO: move to visutils
    """
    Normalizes all timeseries to the same length (by interpolating missing values) and plots the mean and standard error
    of the mean of all timeseries.

    :param timeseries: list of pd.Series objects, indexed by time (float, ms) and containing the values to plot
    :param ax: the axes object to plot on

    keyword arguments:
        - interpolation_kind: the kind of interpolation to use for missing values (default: 'linear')

        - primary_color: the color of the mean line and the fill (default: '#034e7b', dark blue)
        - primary_linewidth: the width of the mean line (default: 2)
        - data_label: the label of the mean line (default: '')

        - show_peak: whether to show a vertical line at the peak value (default: False)
        - peak_color: the color of the peak line (default: '#000000', black)
        - peak_linewidth: the width of the peak line (default: 1)

        - show_individual: whether to show the individual timeseries in the background (default: False)
        - secondary_color: the color of the individual timeseries (default: '#a6bddb', light blue)
        - secondary_linewidth: the width of the individual timeseries (default: 1)

        - title_size: the size of the title (default: 14)
        - title: the title (default: "")
        - label_size: the size of the x and y labels (default: 12)
        - xlabel: the x label (default: "Relative Time (%)")
        - ylabel: the y label (default: "")
        - text_size: the size of the tick labels and legend (default: 10)
        - show_legend: whether to show the legend (default: True)
        - legend_location: the location of the legend (default: 'upper right')
    """
    timeseries_df = tsutils.interpolate_and_merge_timeseries(timeseries, interpolation_kind=kwargs.get('interpolation_kind', 'linear'))
    mean = timeseries_df.mean(axis=1)
    sem = timeseries_df.sem(axis=1)

    primary_color = kwargs.get('primary_color', '#034e7b')  # default: dark blue
    primary_linewidth = kwargs.get('primary_linewidth', 2)
    data_label = kwargs.get('data_label', '')
    ax.plot(mean.index, mean, color=primary_color, linewidth=primary_linewidth, label=data_label, zorder=10)
    ax.fill_between(mean.index, mean - sem, mean + sem, color=primary_color, alpha=0.5, zorder=2)

    if kwargs.get('show_peak', False):
        peak_color = kwargs.get('peak_color', '#000000')  # default: black
        peak_linewidth = kwargs.get('peak_linewidth', 1)
        peak_idx = np.argmax(mean)
        ax.axvline(x=mean.index[peak_idx], color=peak_color, linewidth=peak_linewidth,
                   linestyle='--', zorder=10, label="Max Value")

    if kwargs.get('show_individual', False):
        secondary_color = kwargs.get('secondary_color', '#a6bddb')  # default: light blue
        secondary_linewidth = kwargs.get('secondary_linewidth', 1)
        for col in timeseries_df.columns:
            ax.plot(timeseries_df.index, timeseries_df[col], color=secondary_color,
                    linewidth=secondary_linewidth, alpha=0.75, zorder=1)

    label_size = kwargs.get('label_size', 12)
    text_size = kwargs.get('text_size', 10)
    ax.set_xlabel(kwargs.get('xlabel', "Relative Time (%)"), fontsize=label_size)
    ax.set_ylabel(kwargs.get('ylabel', ""), fontsize=label_size)
    ax.set_xticks(ticks=np.arange(0, 110, 10), labels=np.arange(0, 110, 10))
    ax.tick_params(axis='both', which='major', labelsize=text_size)
    ax.set_title(kwargs.get('title', ""), fontsize=kwargs.get('title_size', 14))
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(bottom=max([mean.min() - 2 * sem.max(), y_bottom]), top=min([mean.max() + 2 * sem.max(), y_top]))
    if kwargs.get('show_legend', True):
        ax.legend(loc=kwargs.get('legend_location', 'upper right'), fontsize=text_size)
    return ax

