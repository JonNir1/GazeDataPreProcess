import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from typing import Tuple, List, Union, Optional


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


def set_figure_properties(fig: Optional[plt.Figure], **kwargs) -> plt.Figure:
    """
    Sets properties of the given figure if provided, otherwise creates a new figure and sets its properties.
    :param fig: existing figure to set properties on, or None to create a new figure.

    keyword arguments:
        - figsize: tuple of floats (width, height) in inches. default: (16, 9).
        - figure_dpi: the DPI of the figure. default: 500.
        - title: the title of the figure. default: "".
        - title_size: the font size of the figure title. default: 18.
        - title_height: the height of the figure title. default: 0.95.
        - tight_layout: whether to call fig.tight_layout(). default: True.

    :return: the figure with the given properties.
    """
    fig = fig if fig is not None else plt.Figure()
    figsize = kwargs.get('figsize', (16, 9))
    fig.set_size_inches(w=figsize[0], h=figsize[1])
    fig.set_dpi(kwargs.get('figure_dpi', 500))
    default_title_height = 1 if len(fig.axes) <= 1 else 0.95
    fig.suptitle(t=kwargs.get('title', ''), fontsize=kwargs.get('title_size', 18),
                 y=kwargs.get("title_height", default_title_height))
    if kwargs.get('tight_layout', True):
        fig.tight_layout()
    return fig


def set_axes_properties(ax: plt.Axes, **kwargs) -> plt.Axes:
    """
    Sets properties of the given axis, such as title, labels, ticks, etc.
    :param ax: plt.Axes object to set properties on.

    keyword arguments:
        - ax_title: the title of the axis. default: ""
        - subtitle_size: the font size of the axis title. default: 14.
        - xlabel: the label of the x-axis. default: ""
        - ylabel: the label of the y-axis. default: ""
        - label_size: the font size of the axis labels. default: 12.
        - show_legend: whether to show the legend. default: False.
        - legend_location: the location of the legend. default: "upper right".
        - hide_axes: whether to hide the axes. default: False.
        - text_size: the font size of the axis ticks. default: 10.
        - invert_yaxis: whether to invert the y-axis. default: False.
    """
    if kwargs.get("ax_title", ""):
        ax.set_title(kwargs.pop("ax_title", ""), fontsize=kwargs.get("subtitle_size", 14))
    if kwargs.get("xlabel", ""):
        ax.set_xlabel(kwargs.pop("xlabel", ""), fontsize=kwargs.get("label_size", 12))
    if kwargs.get("ylabel", ""):
        ax.set_ylabel(kwargs.pop("ylabel", ""), fontsize=kwargs.get("label_size", 12))

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


def generic_bar_chart(ax: plt.Axes,
                      centers: List[np.ndarray], values: [np.ndarray],
                      bar_width: float, **kwargs) -> plt.Axes:
    """
    Plots a bar chart on the given axis.
    :param ax: the axis to plot the bar chart on.
    :param centers: list of numpy arrays, each array contains the center of the bars for one dataset.
    :param values: list of numpy arrays, each array contains the values of the bars for one dataset.
    :param bar_width: the width of all bars.

    keyword arguments:
        - data_labels: A list of labels for the datasets. If specified, must be of the same length as the centers/values lists.
        - cmap: The colormap to use for the bars. default: plt.cm.get_cmap("tab20").
        - alpha: The alpha value of the bars. default: 0.8.

    :raises ValueError: if the length of the centers/values lists is not equal.
            ValueError: if the bar width is invalid.
            ValueError: if the number of labels is not zero or equal to the number of centers/values lists.
    """
    # verify inputs:
    if len(centers) != len(values):
        raise ValueError(f"Length of centers ({len(centers)}) must be equal to length of values ({len(values)})!")
    if not np.isfinite(bar_width) or (bar_width <= 0):
        raise ValueError(f"Invalid bar width ({bar_width})!")
    data_labels = kwargs.get("data_labels", [])
    if (len(data_labels) != len(centers)) or (len(data_labels) == 0):
        raise ValueError(f"Number of labels ({len(data_labels)}) must be equal to number of datasets ({len(centers)})!")

    # plot the distributions:
    cmap_name = kwargs.get("cmap", "tab20")
    alpha = kwargs.get("alpha", 0.8)
    for i, (c, v) in enumerate(zip(centers, values)):
        label = data_labels[i] if len(data_labels) > 0 else None
        edgecolor = get_rgba_color(color=2 * i, cmap_name=cmap_name)
        facecolor = get_rgba_color(color=2 * i + 1, cmap_name=cmap_name)
        ax.bar(c, v, width=bar_width, label=label, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    return ax


def generic_line_chart(ax: plt.Axes,
                       xs: List[np.ndarray], ys: List[np.ndarray],
                       **kwargs) -> plt.Axes:
    """
    Plots multiple lines on the given axis.
    :param ax: the axis to plot the lines on.
    :param xs: list of numpy arrays, each array contains the x-values of the line for one dataset.
    :param ys: list of numpy arrays, each array contains the y-values of the line for one dataset.

    keyword arguments:
        - data_labels: A list of labels for the datasets. If specified, must be of the same length as the xs/ys lists.
        - sems: A list of SEMs for the datasets. If specified, must be of the same length as the xs/ys lists.
        - cmap: The colormap to use for the lines. default: plt.cm.get_cmap("tab20").
        - ls/line_style/linestyle: The style of the primary line. default: "-".
        - lw/line_width/linewidth: The width of the primary line. default: 2.
        - show_peak: Whether to show the peak of each line. default: False.
        - text_size: The size of the axis label and ticks. default: 10.

    :raises ValueError: if the length of the xs/ys lists is not equal.
            ValueError: if the number of labels is not zero or equal to the number of xs/ys lists.
            ValueError: if the number of SEMs is not zero or equal to the number of xs/ys lists.
    """
    # verify inputs:
    if len(xs) != len(ys):
        raise ValueError(f"Number of x-arrays ({len(xs)}) must be equal to number of y-arrays ({len(ys)})!")
    data_labels = kwargs.get("data_labels", [])
    if (len(data_labels) != len(xs)) and (len(data_labels) != 0):
        raise ValueError(f"Number of labels ({len(data_labels)}) must be equal to number of datasets ({len(xs)})!")
    sems = kwargs.get("sems", [])
    if (len(sems) != len(xs)) and (len(sems) != 0):
        raise ValueError(f"Number of SEMs ({len(sems)}) must be equal to number of datasets ({len(xs)})!")

    # plot the lines:
    cmap_name = kwargs.get("cmap", "tab20")
    linestyle = kwargs.get("ls", None) or kwargs.get("line_style", None) or kwargs.get("linestyle", "-")
    primary_line_width = kwargs.get("lw", None) or kwargs.get("line_width", None) or kwargs.get("linewidth", 2)
    secondary_line_width = max(1, primary_line_width // 2)
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for i, (x, y) in enumerate(zip(xs, ys)):
        label = data_labels[i] if len(data_labels) > 0 else None
        color = get_rgba_color(color=2*i, cmap_name=cmap_name)
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=primary_line_width, zorder=i)

        # calculate the min/max values:
        try:
            min_x = min(min_x, np.nanmin(x))
            max_x = max(max_x, np.nanmax(x))
            min_y = min(min_y, np.nanmin(y - sems[i])) if len(sems) > 0 else min(min_y, np.nanmin(y))
            max_y = max(max_y, np.nanmax(y + sems[i])) if len(sems) > 0 else max(max_y, np.nanmax(y))
        except ValueError:
            pass

        # plot the SEMs:
        if len(sems) > 0:
            ax.plot(x, y - sems[i], color=color, linewidth=secondary_line_width, alpha=0.4, zorder=i)
            ax.plot(x, y + sems[i], color=color, linewidth=secondary_line_width, alpha=0.4, zorder=i)
            ax.fill_between(x, y - sems[i], y + sems[i], color=color, alpha=0.2, zorder=i)

    # add vertical lines to mark peaks:
    if kwargs.get("show_peak", False):
        ymin, ymax = ax.get_ylim()
        peak_idxs = [np.argmax(y) for y in ys]
        peak_xs = [xs[i][peak_idxs[i]] for i in range(len(xs))]
        peak_colors = [get_rgba_color(color=2*i, cmap_name=cmap_name) for i in range(len(xs))]
        ax.vlines(x=peak_xs, ymin=ymin, ymax=ymax, color=peak_colors, lw=secondary_line_width, ls='--')

    # set axis ticks and labels:
    set_axis_ticks(ax=ax, min_val=min_x, max_val=max_x, axis='x', **kwargs)
    set_axis_ticks(ax=ax, min_val=min_y, max_val=max_y, axis='y', **kwargs)
    return ax


def set_axis_ticks(ax, min_val: float, max_val: float, axis: str, **kwargs):
    """
    Sets the ticks of the given axis to the given values.
    :param ax: The plt.Axes object.
    :param min_val: The minimum value.
    :param max_val: The maximum value.
    :param axis: The axis to set the ticks for. Must be either 'x' or 'y'.
    :param kwargs: Additional keyword arguments to pass to ax.set_xticks or ax.set_yticks.
    :raises ValueError: If axis is not 'x' or 'y'.
    """
    axis = axis.lower()
    if axis not in ['x', 'y']:
        raise ValueError(f"Invalid axis '{axis}'! Must be either 'x' or 'y'.")
    scale = round(float(np.nan_to_num(np.log10(max_val - min_val) - 1, nan=0)))
    jumps = 2 * np.power(10., scale)
    ticks = np.arange(min_val, max_val + jumps / 2, jumps)
    labels = [np.round(tck, 1 - scale) for tck in ticks]
    if axis == 'x':
        ax.set_xticks(ticks=ticks, labels=labels,
                      fontsize=kwargs.get("text_size", 10), rotation=kwargs.get("rotation", 45))
    else:
        ax.set_yticks(ticks=ticks, labels=labels,
                      fontsize=kwargs.get("text_size", 10), rotation=kwargs.get("rotation", 45))
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
