import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import Utils.timeseries_utils as tsutils
import Visualization.visualization_utils as visutils
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent
from GazeEvents.FixationEvent import FixationEvent


def pupil_size_profile(fixations: List[FixationEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    pupil_sizes = [f.get_pupil_series() for f in fixations]
    kwargs['ylabel'] = "Pupil Size (mm)"
    kwargs['title'] = kwargs.get('title', "Pupil Size Dynamics")
    ax = dynamic_profile(ax=ax, datasets=[pupil_sizes], **kwargs)
    return ax


def velocity_profile(events: List[BaseVisualGazeEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    velocities = [e.get_velocity_series() for e in events]
    kwargs['ylabel'] = "Velocity (px/s)"
    kwargs['title'] = kwargs.get('title', "Velocity Dynamics")
    ax = dynamic_profile(ax=ax, datasets=[velocities], **kwargs)
    return ax


def acceleration_profile(events: List[BaseVisualGazeEvent], ax: plt.Axes, **kwargs) -> plt.Axes:
    # TODO: calculate accelerations and plot them
    kwargs['ylabel'] = "Acceleration (px/s^2)"
    kwargs['title'] = kwargs.get('title', "Acceleration Dynamics")
    raise NotImplementedError


def dynamic_profile(ax: plt.Axes, datasets: List[List[pd.Series]], **kwargs) -> plt.Axes:
    """
    Coerces all datasets to have time range of [0, 1] and within each dataset, interpolates all series to have the same
    number of samples, then plots the mean and standard error of each dataset on the given axis.

    :param ax: axis to plot on
    :param datasets: list of datasets, each dataset is a list of pd.Series where the indexes are timestamps (floats, ms)
                        and the values are the measured samples of the time series.

    keyword arguments:
        - interpolation_kind: The kind of interpolation to perform (default: 'linear')
        - show_sems: whether to show the standard error of the mean (default: True)

        keywords for generic_line_chart():
        - data_labels: A list of labels for the datasets. If specified, must be of the same length as the datasets list.
        - show_peak: whether to mark the peak of the dynamics with a vertical line (default: True)
        - cmap: The colormap to use for the bars. default: plt.cm.get_cmap("tab20").
        - lw/line_width/linewidth: The width of the plotted lines. default: 2.
        - show_peak: Whether to show the peak of each line. default: False.

        keywords for set_axes_properties():
        - title: The title of the axes.
        - title_size: The size of the title. default: 14.
        - xlabel: The label of the x-axis.
        - ylabel: The label of the y-axis. default: "".
        - text_size: The size of the axis labels. default: 12.
        - show_legend: Whether to show the legend. default: False.
        - legend_location: The location of the legend. default: "upper right".

    :return:
    """
    # calculate mean and sem for each dataset:
    means: List[pd.Series] = []
    sems: List[pd.Series] = []
    interpolation_kind = kwargs.get('interpolation_kind', 'linear')
    show_sems = kwargs.get('show_sems', True)
    for dataset in datasets:
        interpolated_df: pd.DataFrame = tsutils.interpolate_and_merge_timeseries(dataset, interpolation_kind)
        means.append(interpolated_df.mean(axis=1))
        if show_sems:
            sems.append(interpolated_df.sem(axis=1))

    # plot:
    kwargs["show_peak"] = kwargs.get("show_peak", True)  # mark peak of dynamics with a vertical line (default: True)
    ax = visutils.generic_line_chart(ax=ax,
                                     xs=[m.index.to_numpy() for m in means],
                                     ys=[m.values for m in means],
                                     sems=[s.values for s in sems],
                                     **kwargs)
    # set axes properties:
    visutils.set_axes_properties(ax=ax, ax_title=kwargs.pop("title", "Dynamics"),
                                 subtitle_size=kwargs.pop("title_size", 14), ylabel=kwargs.pop("ylabel", "%"),
                                 **kwargs)
    return ax

