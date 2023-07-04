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
    # TODO: control tick values and labels
    if kwargs.get('invert_yaxis', False):
        # invert y-axis to match the screen coordinates:
        ax.invert_yaxis()
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

