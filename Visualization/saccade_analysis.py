import numpy as np
from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.SaccadeEvent import SaccadeEvent


def histograms_figure(saccades: List[SaccadeEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        saccades = [s for s in saccades if not s.is_outlier]

    fig = plt.Figure()
    visutils.set_figure_properties(fig, title=kwargs.pop("title", "Saccade Summary"),
                                   figsize=kwargs.pop("figsize", (21, 14)), **kwargs)
    nbins = kwargs.get("nbins", 20)
    edge_color = visutils.get_rgba_color(color=0, cmap_name=kwargs.get("cmap_name", "tab20"))
    face_color = visutils.get_rgba_color(color=1, cmap_name=kwargs.get("cmap_name", "tab20"))

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist([s.duration for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax1, ax_title="Durations", xlabel="Duration (ms)", ylabel="Counts", **kwargs)

    # max velocity histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist([s.max_velocity for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax2, ax_title="Maximum Velocities", xlabel="Velocity (px / s)", ylabel="Counts", **kwargs)

    # amplitude histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist([s.amplitude for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax3, ax_title="Amplitude", xlabel="Amplitude (°)", ylabel="Counts", **kwargs)

    # azimuth counts (polar)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    __create_rose_plot(data=[s.azimuth for s in saccades], ax=ax4, nbins=nbins,
                       face_color=face_color, edge_color=edge_color)
    visutils.set_axes_texts(ax4, ax_title="Azimuth (°)", xlabel="", ylabel="", **kwargs)
    return fig


def __create_rose_plot(data, ax: plt.Axes, nbins: int, face_color, edge_color,
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
