import numpy as np
from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
import Visualization.distributions as distributions
from GazeEvents.FixationEvent import FixationEvent


def distributions_figure(fixations: List[FixationEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    fig = visutils.set_figure_properties(fig=None, title=kwargs.pop("title", "Fixation Summary"),
                                         figsize=kwargs.pop("figsize", (21, 14)), **kwargs)

    # durations distribution
    ax1 = fig.add_subplot(2, 2, 1)
    durations_data = [np.array([f.duration for f in fixations])]
    distributions.bar_chart(ax=ax1, datasets=durations_data,
                            data_labels=["All Fixations"], title="Durations (ms)", **kwargs)

    # dispersion distribution
    ax2 = fig.add_subplot(2, 2, 2)
    dispersions_data = [np.array([f.max_dispersion for f in fixations])]
    distributions.bar_chart(ax=ax2, datasets=dispersions_data,
                            data_labels=["All Fixations"], title="Dispersions (px)", **kwargs)

    # max velocity distribution
    ax3 = fig.add_subplot(2, 2, 3)
    max_velocities_data = [np.array([f.max_velocity for f in fixations])]
    distributions.bar_chart(ax=ax3, datasets=max_velocities_data,
                            data_labels=["All Fixations"], title="Maximum Velocities (px/s)",
                            **kwargs)

    # mean velocity distribution
    ax4 = fig.add_subplot(2, 2, 4)
    mean_velocities_data = [np.array([f.mean_velocity for f in fixations])]
    distributions.bar_chart(ax=ax4, datasets=mean_velocities_data,
                            data_labels=["All Fixations"], title="Mean Velocities (px/s)",
                            **kwargs)

    # mean pupil size distribution
    # TODO: create histogram for pupil size
    return fig
