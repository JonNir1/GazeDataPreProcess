import numpy as np
import matplotlib.pyplot as plt
from typing import List

import Utils.array_utils as au
import Visualization.visualization_utils as visutils


def bar_chart(ax: plt.Axes, datasets: List[np.ndarray], **kwargs) -> plt.Axes:
    """
    Calculates the distribution of each dataset and plots a bar chart of the distributions on the given axis.

    :param ax: The axis to plot the distributions on.
    :param datasets: A list of numpy arrays, each containing the data of a distribution.

    keyword arguments:
        - nbins: The number of bins to use for the histogram.
        - min_percentage_threshold: The minimum percentage of data points in a bin for it to be included in the
                                    distribution (bins with less data points will be ignored). default: 1.
        - labels: A list of labels for the datasets. If specified, must be of the same length as the datasets list.
        - cmap: The colormap to use for the bars. default: plt.cm.get_cmap("tab20").
        - title: The title of the axes.
        - title_size: The size of the title. default: 14.
        - xlabel: The label of the x-axis.
        - ylabel: The label of the y-axis. default: "%".
        - text_size: The size of the axis labels. default: 12.
        - show_legend: Whether to show the legend. default: False.
        - legend_location: The location of the legend. default: "upper right".
    """
    # extract the distributions:
    nbins = kwargs.get("nbins", 20)
    min_percentage_threshold = kwargs.get("min_percentage_threshold", 1)
    percentages, centers = [], []
    for data in datasets:
        p, c = au.calculate_distribution(data, nbins=nbins, min_threshold=min_percentage_threshold)
        percentages.append(p)
        centers.append(c)

    # plot the distributions:
    labels = kwargs.get("labels", [])
    if (len(labels) != len(datasets)) or (len(labels) == 0):
        raise ValueError(f"Number of labels ({len(labels)}) must be equal to number of datasets ({len(datasets)})!")
    cmap_name = kwargs.get("cmap", "tab20")
    bar_width = min([np.min(np.diff(c)) for c in centers]) * 0.9
    for i, (p, c) in enumerate(zip(percentages, centers)):
        edgecolor = visutils.get_rgba_color(color=2 * i, cmap_name=cmap_name)
        facecolor = visutils.get_rgba_color(color=2 * i + 1, cmap_name=cmap_name)
        ax.bar(c, p, width=bar_width, label=labels[i], facecolor=facecolor, edgecolor=edgecolor, alpha=0.8)

    # set the axis properties:
    visutils.set_axes_texts(ax=ax, ax_title=kwargs.get("title", ""),
                            xlabel=kwargs.get("xlabel", ""), ylabel=kwargs.get("ylabel", "%"), **kwargs)
    return ax
