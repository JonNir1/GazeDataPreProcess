import numpy as np
from typing import List
import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
import Visualization.distributions as distributions
from GazeEvents.SaccadeEvent import SaccadeEvent


def distributions_figure(saccades: List[SaccadeEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        saccades = [s for s in saccades if not s.is_outlier]
    fig = visutils.set_figure_properties(fig=None, title=kwargs.pop("title", "Saccade Summary"),
                                         figsize=kwargs.pop("figsize", (21, 14)), **kwargs)

    # TODO: add main sequence plot

    # durations distribution
    ax1 = fig.add_subplot(2, 2, 1)
    durations_data = [np.array([s.duration for s in saccades])]
    distributions.bar_chart(ax=ax1, datasets=durations_data,
                            data_labels=["All Saccades"], title="Durations (ms)", **kwargs)

    # max velocity distribution
    ax2 = fig.add_subplot(2, 2, 2)
    max_velocities_data = [np.array([s.max_velocity for s in saccades])]
    distributions.bar_chart(ax=ax2, datasets=max_velocities_data,
                            data_labels=["All Saccades"], title="Maximum Velocities (px/s)",
                            **kwargs)

    # amplitude distribution
    ax3 = fig.add_subplot(2, 2, 3)
    amplitude_data = [np.array([s.amplitude for s in saccades if np.isfinite(s.amplitude)])]
    distributions.bar_chart(ax=ax3, datasets=amplitude_data,
                            data_labels=["All Saccades"], title="Amplitude (°)", **kwargs)

    # azimuth distribution (polar)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    azimuth_data = [np.array([s.azimuth for s in saccades])]
    distributions.rose_chart(ax=ax4, datasets=azimuth_data, data_labels=["All Saccades"], title="Azimuth (°)", **kwargs)
    return fig

