import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from typing import Tuple, List, Union, Optional

import Utils.array_utils as au


def show_figure(fig):
    """
    Create a dummy figure and use its manager to display the given figure.
    See https://stackoverflow.com/a/54579616/8543025 for more details.
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    fig.show()


def save_figure(fig: plt.Figure, full_path: str, **kwargs):
    """
    Save figure to given path.
    """
    dpi = kwargs.get("dpi", "figure")
    bbox_inches = kwargs.get("bbox_inches", "tight")
    is_transparent = kwargs.get("transparent", False)
    fig.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches, transparent=is_transparent)


def get_rgba_color(color: Union[str, int], cmap_name: Optional[str]) -> Tuple[float, float, float, float]:
    """
    Returns the color to use based on the color and cmap arguments.
    If `cmap_name` is None, `color` must be a string representing a color. Otherwise, `color` must be an integer
    representing the index of the color in the given cmap.

    :return the matplotlib RGBA color: tuple of floats in range [0,1]

    :raise ValueError: If `cmap_name` is None and `color` is not a string representing a color.
    """
    if cmap_name is None:
        if isinstance(color, str):
            return mcolors.to_rgba(color)
        raise ValueError(f"Invalid color '{color}'! Must be a string representing a color.")
    cmap = plt.colormaps.get_cmap(cmap_name)
    return cmap(color)


def set_figure_properties(fig: plt.Figure, **kwargs) -> plt.Figure:
    figsize = kwargs.get('figsize', (16, 9))
    fig.set_size_inches(w=figsize[0], h=figsize[1])
    fig.set_dpi(kwargs.get('figure_dpi', 500))
    fig.suptitle(t=kwargs.get('title', ''), fontsize=kwargs.get('title_size', 18), y=kwargs.get("title_height", 0.95))
    if kwargs.get('tight_layout', True):
        fig.tight_layout()
    return fig


def set_axes_texts(ax: plt.Axes, ax_title: Optional[str],
                   xlabel: Optional[str], ylabel: Optional[str],
                   **kwargs) -> plt.Axes:
    if ax_title:
        ax.set_title(ax_title, fontsize=kwargs.get("subtitle_size", 16))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=kwargs.get("label_size", 12))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=kwargs.get("label_size", 12))

    text_size = kwargs.get("text_size", 10)
    if kwargs.get("show_legend", False):
        ax.legend(loc=kwargs.get("legend_location", "upper right"), fontsize=text_size)
    if kwargs.get("hide_axes", False):
        ax.axis("off")
        return ax
    ax.tick_params(axis='both', which='major', labelsize=text_size)
    if kwargs.get('invert_yaxis', False):
        # invert y-axis to match the screen coordinates:
        ax.invert_yaxis()
    return ax


def create_rose_plot(data, ax: plt.Axes, nbins: int, face_color, edge_color,
                     zero_location: str = "E", clockwise_angles: bool = False) -> plt.Axes:
    if ax.name != "polar":
        raise ValueError(f"Invalid axis type '{ax.name}'! Must be a polar axis.")
    counts, _edges = np.histogram(data, bins=np.arange(0, 361, 360 / nbins))
    n = len(counts)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = (2 * np.pi) / n
    _bars = ax.bar(angles, counts, width=width, bottom=0.0, color=face_color, edgecolor=edge_color)
    ax.set_theta_zero_location(zero_location)  # set 0° to the provided location (default: East)
    if clockwise_angles:
        ax.set_theta_direction(-1)  # set the direction of the angles to clockwise
    return ax


def distribution_comparison(ax: plt.Axes, datasets: List[np.ndarray], **kwargs) -> plt.Axes:
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
        edgecolor = get_rgba_color(color=2 * i, cmap_name=cmap_name)
        facecolor = get_rgba_color(color=2 * i + 1, cmap_name=cmap_name)
        ax.bar(c, p, width=bar_width, label=labels[i], facecolor=facecolor, edgecolor=edgecolor, alpha=0.8)

    # set the axis properties:
    set_axes_texts(ax=ax, ax_title=kwargs.get("title", ""),
                   xlabel=kwargs.get("xlabel", ""), ylabel=kwargs.get("ylabel", "%"), **kwargs)
    return ax


def get_line_axis_limits(ax: plt.Axes, axis: str) -> Tuple[float, float]:
    """
    Returns the maximun and minimum values among all lines in the given plt.Axes object.
    :raises ValueError: If axis is not 'x' or 'y'.
    """
    axis = axis.lower()
    if axis not in ['x', 'y']:
        raise ValueError(f"Invalid axis '{axis}'! Must be either 'x' or 'y'.")
    axis_idx = 0 if axis == 'x' else 1
    min_val, max_val = float('inf'), float('-inf')
    for ln in ax.lines:
        data = ln.get_data()[axis_idx]
        tmp_min = float(np.min(np.ma.masked_invalid(data)))
        tmp_max = float(np.max(np.ma.masked_invalid(data)))
        min_val = min(min_val, tmp_min)
        max_val = max(max_val, tmp_max)
    return min_val, max_val


def set_line_axis_ticks_and_limits(ax: plt.Axes, axis: str, text_size: int) -> plt.Axes:
    """
    Sets the X/Y axis limits and ticks for the given plt.Axes object, based on the lines in it.
    :param ax: The plt.Axes object to set the properties of.
    :param axis: The axis to set the properties of. Must be either 'x' or 'y'.
    :param text_size: The size of the axis label and ticks.

    :raises ValueError: If axis is not 'x' or 'y'.
    """
    axis = axis.lower()
    if axis not in ['x', 'y']:
        raise ValueError(f"Invalid axis '{axis}'! Must be either 'x' or 'y'.")

    _, axis_max = get_line_axis_limits(ax, axis=axis)
    jumps = 2 * np.power(10, max(0, int(np.log10(axis_max)) - 1))
    if axis == 'x':
        ax.set_xlim(left=int(-0.02 * axis_max), right=int(1.05 * axis_max))
        xticks = [int(val) for val in np.arange(int(axis_max)) if val % jumps == 0]
        ax.set_xticks(ticks=xticks, labels=[str(tck) for tck in xticks], fontsize=text_size)

    if axis == 'y':
        ax.set_ylim(bottom=int(-0.02 * axis_max), top=int(1.05 * axis_max))
        yticks = [int(val) for val in np.arange(int(axis_max)) if val % jumps == 0]
        ax.set_yticks(ticks=yticks, labels=[str(tck) for tck in yticks], fontsize=text_size)
    return ax

