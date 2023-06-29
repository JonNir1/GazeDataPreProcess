from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.FixationEvent import FixationEvent


def fixation_histograms_figure(fixations: List[FixationEvent], ignore_outliers: bool = True, **kwargs):
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig, axes = plt.subplots(2, 2, figsize=kwargs.get("figsize", (21, 14)))
    fig.suptitle(t=kwargs.get("title", "Fixation Summary"), font_size=kwargs.get("title_size", 16), y=0.98)
    nbins = kwargs.get("nbins", 20)
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    ax = visutils.create_histogram([f.duration for f in fixations], axes[0, 0], title="Durations",
                                   xlabel="Duration (ms)", color=kwargs.get("duration_color", "lightblue"),
                                   nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # dispersion histogram
    ax = visutils.create_histogram([f.max_dispersion for f in fixations], axes[0, 1], title="Dispersions",
                                   xlabel="Dispersion (pixels)", color=kwargs.get("dispersion_color", "lightblue"),
                                   nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    ax = visutils.create_histogram([f.max_velocity for f in fixations], axes[1, 0], title="Max Velocities",
                                   xlabel="Max Velocity (pixels/ms)", color=kwargs.get("max_vel_color", "lightblue"),
                                   nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # mean velocity histogram
    ax = visutils.create_histogram([f.mean_velocity for f in fixations], axes[1, 1], title="Mean Velocities",
                                   xlabel="Mean Velocity (pixels/ms)", color=kwargs.get("mean_vel_color", "lightblue"),
                                   nbins=nbins, title_size=subtitle_size, label_size=label_size)
    return fig
