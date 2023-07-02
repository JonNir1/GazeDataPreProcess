import numpy as np
from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.SaccadeEvent import SaccadeEvent


def saccade_histograms_figure(saccades: List[SaccadeEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        saccades = [s for s in saccades if not s.is_outlier]

    fig = plt.Figure(figsize=kwargs.get("figsize", (21, 14)))
    fig.suptitle(t=kwargs.get("title", "Saccade Summary"), fontsize=kwargs.get("title_size", 16), y=0.95)
    nbins = kwargs.get("nbins", 20)
    face_color = kwargs.get("face_color", "lightblue")
    edge_color = kwargs.get("edge_color", "darkblue")
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    visutils.create_histogram([s.duration for s in saccades], ax1, title="Durations",
                              xlabel="Duration (ms)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax2 = fig.add_subplot(2, 2, 2)
    visutils.create_histogram([s.max_velocity for s in saccades], ax2, title="Maximum Velocities",
                              xlabel="Velocity (px / s)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # amplitude histogram
    ax3 = fig.add_subplot(2, 2, 3)
    visutils.create_histogram([s.amplitude for s in saccades], ax3, title="Amplitude",
                              xlabel="Amplitude (°)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # azimuth counts (polar)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    azimuths = [s.azimuth for s in saccades if not np.isnan(s.azimuth)]
    counts, _edges = np.histogram(azimuths, bins=np.arange(0, 361, 360 / nbins))
    visutils.create_rose_plot(counts, ax4, title="Azimuth (°)", xlabel="", ylabel="",
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)
    return fig
