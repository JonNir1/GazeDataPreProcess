from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.FixationEvent import FixationEvent


def fixation_histograms_figure(fixations: List[FixationEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig = plt.Figure(figsize=kwargs.get("figsize", (21, 14)))
    fig.suptitle(t=kwargs.get("title", "Fixation Summary"), fontsize=kwargs.get("title_size", 16), y=0.95)
    nbins = kwargs.get("nbins", 20)
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    visutils.create_histogram([f.duration for f in fixations], ax1, title="Durations",
                              xlabel="Duration (ms)", color=kwargs.get("duration_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # dispersion histogram
    ax2 = fig.add_subplot(2, 2, 2)
    visutils.create_histogram([f.max_dispersion for f in fixations], ax2, title="Dispersions",
                              xlabel="Dispersion (pixels)", color=kwargs.get("dispersion_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax3 = fig.add_subplot(2, 2, 3)
    visutils.create_histogram([f.max_velocity for f in fixations], ax3, title="Maximum Velocities",
                              xlabel="Max Velocity (pixels/ms)", color=kwargs.get("max_vel_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # mean velocity histogram
    ax4 = fig.add_subplot(2, 2, 4)
    visutils.create_histogram([f.mean_velocity for f in fixations], ax4, title="Mean Velocities",
                              xlabel="Mean Velocity (pixels/ms)", color=kwargs.get("mean_vel_color", "lightblue"),
                              nbins=nbins, title_size=subtitle_size, label_size=label_size)
    return fig
