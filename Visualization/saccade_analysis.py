import numpy as np
from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.SaccadeEvent import SaccadeEvent


def histograms_figure(saccades: List[SaccadeEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        saccades = [s for s in saccades if not s.is_outlier]

    fig = plt.Figure(figsize=kwargs.get("figsize", (21, 14)))
    fig.suptitle(t=kwargs.get("title", "Saccade Summary"), fontsize=kwargs.get("title_size", 16), y=0.95)
    nbins = kwargs.get("nbins", 20)
    edge_color = visutils.get_rgba_color(color=0, cmap_name=kwargs.get("cmap_name", "tab20"))
    face_color = visutils.get_rgba_color(color=1, cmap_name=kwargs.get("cmap_name", "tab20"))
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist([s.duration for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax1, title="Durations", xlabel="Duration (ms)", ylabel="Counts",
                            title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist([s.max_velocity for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax2, title="Maximum Velocities", xlabel="Velocity (px / s)", ylabel="Counts",
                            title_size=subtitle_size, label_size=label_size)

    # amplitude histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist([s.amplitude for s in saccades], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax3, title="Amplitude", xlabel="Amplitude (°)", ylabel="Counts",
                            title_size=subtitle_size, label_size=label_size)

    # azimuth counts (polar)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    visutils.create_rose_plot(data=[s.azimuth for s in saccades], ax=ax4, nbins=nbins,
                              face_color=face_color, edge_color=edge_color)
    visutils.set_axes_texts(ax4, title="Azimuth (°)", xlabel="", ylabel="",
                            title_size=subtitle_size, label_size=label_size)
    return fig
