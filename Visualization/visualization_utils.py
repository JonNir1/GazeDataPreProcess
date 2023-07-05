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
    fig.suptitle(t=kwargs.get('title', ''), fontsize=kwargs.get('title_size', 18), y=kwargs.get("title_height", 0.95))
    if kwargs.get('tight_layout', True):
        fig.tight_layout()
    return fig


def set_axes_properties(ax: plt.Axes, ax_title: Optional[str],
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
    # TODO: control tick values and labels
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
        - labels: A list of labels for the datasets. If specified, must be of the same length as the centers/values lists.
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
    labels = kwargs.get("labels", [])
    if (len(labels) != len(centers)) or (len(labels) == 0):
        raise ValueError(f"Number of labels ({len(labels)}) must be equal to number of datasets ({len(centers)})!")

    # plot the distributions:
    cmap_name = kwargs.get("cmap", "tab20")
    alpha = kwargs.get("alpha", 0.8)
    for i, (c, v) in enumerate(zip(centers, values)):
        edgecolor = get_rgba_color(color=2 * i, cmap_name=cmap_name)
        facecolor = get_rgba_color(color=2 * i + 1, cmap_name=cmap_name)
        ax.bar(c, v, width=bar_width, label=labels[i], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
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
        - labels: A list of labels for the datasets. If specified, must be of the same length as the xs/ys lists.
        - sems: A list of SEMs for the datasets. If specified, must be of the same length as the xs/ys lists.
        - cmap: The colormap to use for the lines. default: plt.cm.get_cmap("tab20").
        - lw/line_width/linewidth: The width of the primary line. default: 2.
        - show_peak: Whether to show the peak of each line. default: False.

    :raises ValueError: if the length of the xs/ys lists is not equal.
            ValueError: if the number of labels is not zero or equal to the number of xs/ys lists.
            ValueError: if the number of SEMs is not zero or equal to the number of xs/ys lists.
    """
    # verify inputs:
    if len(xs) != len(ys):
        raise ValueError(f"Number of x-arrays ({len(xs)}) must be equal to number of y-arrays ({len(ys)})!")
    labels = kwargs.get("labels", [])
    if (len(labels) != len(xs)) and (len(labels) != 0):
        raise ValueError(f"Number of labels ({len(labels)}) must be equal to number of datasets ({len(xs)})!")
    sems = kwargs.get("sems", [])
    if (len(sems) != len(xs)) and (len(sems) != 0):
        raise ValueError(f"Number of SEMs ({len(sems)}) must be equal to number of datasets ({len(xs)})!")

    # plot the lines:
    cmap_name = kwargs.get("cmap", "tab20")
    primary_line_width = kwargs.get("lw", None) or kwargs.get("line_width", None) or kwargs.get("linewidth", 2)
    secondary_line_width = max(1, primary_line_width // 2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = get_rgba_color(color=2*i, cmap_name=cmap_name)
        ax.plot(x, y, label=labels[i], color=color, linewidth=primary_line_width, zorder=i)
        if len(sems) > 0:
            ax.plot(x, y - sems[i], color=color, linewidth=secondary_line_width, alpha=0.4, zorder=i)
            ax.plot(x, y + sems[i], color=color, linewidth=secondary_line_width, alpha=0.4, zorder=i)
            ax.fill_between(x, y - sems[i], y + sems[i], color=color, alpha=0.2, zorder=i)
        if kwargs.get("show_peak", False):
            peak_idx = np.argmax(y)
            peak_color = get_rgba_color(color=2*i+1, cmap_name=cmap_name)
            ax.vlines(x[peak_idx], ymin=0, ymax=y[peak_idx], color=peak_color, linewidth=secondary_line_width, zorder=i)
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

