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
    face_color = kwargs.get("face_color", "lightblue")
    edge_color = kwargs.get("edge_color", "darkblue")
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    visutils.create_histogram([f.duration for f in fixations], ax1, title="Durations",
                              xlabel="Duration (ms)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # dispersion histogram
    ax2 = fig.add_subplot(2, 2, 2)
    visutils.create_histogram([f.max_dispersion for f in fixations], ax2, title="Dispersions",
                              xlabel="Dispersion (pixels)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax3 = fig.add_subplot(2, 2, 3)
    visutils.create_histogram([f.max_velocity for f in fixations], ax3, title="Maximum Velocities",
                              xlabel="Max Velocity (pixels/ms)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)

    # mean velocity histogram
    ax4 = fig.add_subplot(2, 2, 4)
    visutils.create_histogram([f.mean_velocity for f in fixations], ax4, title="Mean Velocities",
                              xlabel="Mean Velocity (pixels/ms)", ylabel="Counts", nbins=nbins,
                              face_color=face_color, edge_color=edge_color,
                              title_size=subtitle_size, label_size=label_size)
    return fig
