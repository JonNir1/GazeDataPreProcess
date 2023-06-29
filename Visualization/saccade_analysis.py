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
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    visutils.create_histogram([s.duration for s in saccades], ax1, title="Durations",
                              xlabel="Duration (ms)", color=kwargs.get("duration_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax2 = fig.add_subplot(2, 2, 2)
    visutils.create_histogram([s.max_velocity for s in saccades], ax2, title="Maximum Velocities",
                              xlabel="Velocity (px / s)", color=kwargs.get("max_velocity_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # amplitude histogram
    ax3 = fig.add_subplot(2, 2, 3)
    visutils.create_histogram([s.amplitude for s in saccades], ax3, title="Amplitude",
                              xlabel="Amplitude (°)", color=kwargs.get("amplitude_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # azimuth histogram (polar)
    ax4 = fig.add_subplot(2, 2, 4)
    visutils.create_histogram([s.azimuth for s in saccades], ax4, title="Azimuth",
                              xlabel="Azimuth (°)", color=kwargs.get("azimuth_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)
    return fig
