from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from GazeEvents.FixationEvent import FixationEvent


def fixation_histograms_figure(fixations: List[FixationEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig = plt.Figure()
    visutils.set_figure_properties(fig, title=kwargs.pop("title", "Fixation Summary"),
                                   figsize=kwargs.pop("figsize", (21, 14)), **kwargs)
    nbins = kwargs.get("nbins", 20)
    edge_color = visutils.get_rgba_color(color=0, cmap_name=kwargs.get("cmap_name", "tab20"))
    face_color = visutils.get_rgba_color(color=1, cmap_name=kwargs.get("cmap_name", "tab20"))

    # durations histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist([f.duration for f in fixations], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax1, ax_title="Durations",
                            xlabel="Duration (ms)", ylabel="Counts", **kwargs)

    # dispersion histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist([f.max_dispersion for f in fixations], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax2, ax_title="Dispersions",
                            xlabel="Dispersion (pixels)", ylabel="Counts", **kwargs)

    # max velocity histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist([f.max_velocity for f in fixations], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax3, ax_title="Maximum Velocities",
                            xlabel="Max Velocity (pixels/ms)", ylabel="Counts", **kwargs)

    # mean velocity histogram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist([f.mean_velocity for f in fixations], bins=nbins, facecolor=face_color, edgecolor=edge_color)
    visutils.set_axes_texts(ax4, ax_title="Mean Velocities",
                            xlabel="Mean Velocity (pixels/ms)", ylabel="Counts", **kwargs)

    # mean pupil size histogram
    # TODO: create histogram for pupil size
    return fig
