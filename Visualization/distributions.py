import numpy as np
import matplotlib.pyplot as plt
from typing import List

import Utils.array_utils as au
import Visualization.visualization_utils as visutils


def bar_chart(ax: plt.Axes, datasets: List[np.ndarray], **kwargs) -> plt.Axes:
    """
    Calculates the distribution of each dataset and plots a bar chart of the distributions on the given axis.
    The distribution is calculated by splitting the data into `nbins` different bins and calculating the percentage of
    data points in each bin. Bins with less than `min_percentage_threshold` percentage of data points are ignored.
    The bins are then plotted as bars on the x-axis, with width equal to 90% of the minimal difference between the
    bin centers.

    :param ax: The axis to plot the distributions on.
    :param datasets: A list of numpy arrays, each containing the data of a distribution.

    keyword arguments:
        keywords for generic_bar_chart():
        - nbins: The number of bins to use for the histogram.
        - min_percentage_threshold: The minimum percentage of data points in a bin for it to be included in the
                                    distribution (bins with less data points will be ignored). default: 1.
        - data_labels: A list of labels for the datasets. If specified, must be of the same length as the datasets list.
        - cmap: The colormap to use for the bars. default: plt.cm.get_cmap("tab20").

        keywords for set_axes_properties():
        - title: The title of the axes.
        - title_size: The size of the title. default: 14.
        - xlabel: The label of the x-axis.
        - ylabel: The label of the y-axis. default: "%".
        - text_size: The size of the axis labels. default: 12.
        - show_legend: Whether to show the legend. default: False.
        - legend_location: The location of the legend. default: "upper right".
    """
    # calculate the distributions:
    nbins = kwargs.get("nbins", 20)
    min_percentage_threshold = kwargs.get("min_percentage_threshold", 1)
    percentages, centers = [], []
    for data in datasets:
        p, c = au.calculate_distribution(data, nbins=nbins, min_threshold=min_percentage_threshold)
        percentages.append(p)
        centers.append(c)

    # plot the distributions:
    width = min([np.min(np.diff(c)) for c in centers]) * 0.9
    ax = visutils.generic_bar_chart(ax=ax, centers=centers, values=percentages, bar_width=width, **kwargs)

    # set axes properties:
    visutils.set_axes_properties(ax=ax, ax_title=kwargs.pop("title", "Distribution"),
                                 subtitle_size=kwargs.pop("title_size", 14), ylabel=kwargs.pop("ylabel", "%"),
                                 **kwargs)
    return ax


def rose_chart(ax: plt.Axes, datasets: List[np.ndarray], **kwargs) -> plt.Axes:
    """
    Calculates the distribution of each dataset and plots a rose chart of the distributions on the given axis.
    The distribution is calculated by splitting the data into `nbins` different bins and calculating the percentage of
    data points in each bin. The bins are then plotted as bars on a polar axis, with width `360 / nbins`.

    :param ax: The axis to plot the distributions on.
    :param datasets: A list of numpy arrays, each containing the data of a single distribution.

    keyword arguments:
        keywords for generic_bar_chart():
        - nbins: The number of bins to use for the histogram.
        - data_labels: A list of labels for the datasets. If specified, must be of the same length as the datasets list.
        - cmap: The colormap to use for the bars. default: plt.cm.get_cmap("tab20").

        keywords for set_axes_properties():
        - title: The title of the axes.
        - title_size: The size of the title. default: 14.
        - xlabel: The label of the x-axis.
        - ylabel: The label of the y-axis. default: "%".
        - text_size: The size of the axis labels. default: 12.
        - show_legend: Whether to show the legend. default: False.
        - legend_location: The location of the legend. default: "upper right".

        other keywords:
        - zero_location: The location of the 0° angle. default: "E" (East).
        - clockwise_angles: Whether to plot the angles clockwise. default: False.

    :raises ValueError: If the axis is not a polar axis.
    """
    if ax.name != "polar":
        raise ValueError(f"Invalid axis type '{ax.name}'! Must be a polar axis.")

    # calculate the distributions:
    nbins = kwargs.get("nbins", 20)
    percentages = []
    for data in datasets:
        counts, edges = np.histogram(data, bins=np.arange(0, 361, 360 / nbins))
        percentages.append(100 * counts / np.sum(counts))

    # plot the distributions:
    angles = [np.linspace(0, 2 * np.pi, nbins, endpoint=False)]
    width = (2 * np.pi) / nbins
    ax = visutils.generic_bar_chart(ax=ax, centers=angles, values=percentages, bar_width=width, **kwargs)

    # set axes properties:
    visutils.set_axes_properties(ax=ax, ax_title=kwargs.pop("title", "Distribution"),
                                 subtitle_size=kwargs.pop("title_size", 14), ylabel=kwargs.pop("ylabel", "%"),
                                 **kwargs)
    ax.set_theta_zero_location(kwargs.get("zero_location", "E"))  # set 0° to the provided location (default: East)
    if kwargs.get("clockwise_angles", False):
        ax.set_theta_direction(-1)
    return ax

