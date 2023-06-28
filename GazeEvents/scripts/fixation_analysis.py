import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from GazeEvents.FixationEvent import FixationEvent


def fixation_histograms(fixations: List[FixationEvent], ignore_outliers: bool = True, **kwargs):
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig, axes = plt.subplots(2, 2, figsize=kwargs.get("figsize", (21, 14)))
    fig.suptitle("Fixation Summary", y=0.98, font_size=kwargs.get("title_size", 16))

    nbins = kwargs.get("nbins", 20)
    subtitle_size = kwargs.get("subtitle_size", 14)
    label_size = kwargs.get("label_size", 12)

    # durations histogram
    _create_histogram([f.duration for f in fixations], axes[0, 0], title="Fixation Durations",
                      xlabel="Duration (ms)", color=kwargs.get("duration_color", "lightblue"),
                      nbins=nbins, title_size=subtitle_size, label_size=label_size)
    # dispersion histogram
    _create_histogram([f.max_dispersion for f in fixations], axes[0, 1], title="Fixation Dispersions",
                      xlabel="Dispersion (pixels)", color=kwargs.get("dispersion_color", "lightblue"),
                      nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # max velocity histogram
    _create_histogram([f.max_velocity for f in fixations], axes[1, 0], title="Fixation Max Velocities",
                      xlabel="Max Velocity (pixels/ms)", color=kwargs.get("max_vel_color", "lightblue"),
                      nbins=nbins, title_size=subtitle_size, label_size=label_size)

    # mean velocity histogram
    _create_histogram([f.mean_velocity for f in fixations], axes[1, 1], title="Fixation Mean Velocities",
                      xlabel="Mean Velocity (pixels/ms)", color=kwargs.get("mean_vel_color", "lightblue"),
                      nbins=nbins, title_size=subtitle_size, label_size=label_size)
    return fig


def _create_histogram(data, ax: plt.Axes, title: str, xlabel: str,
                      nbins: int, color: str, title_size: int, label_size: int):
    ax.hist(data, bins=nbins, color=color)
    ax.set_title(title, font_size=title_size)
    ax.set_xlabel(xlabel, font_size=label_size)
    ax.set_ylabel("Count", font_size=label_size)
