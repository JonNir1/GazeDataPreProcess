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

    # durations distribution
    ax1 = fig.add_subplot(2, 2, 1)
    durations_data = [np.array([s.duration for s in saccades])]
    distributions.bar_chart(ax=ax1, data=durations_data,
                            labels=["All Saccades"], title="Durations", xlabel="Duration (ms)", **kwargs)

    # max velocity distribution
    ax2 = fig.add_subplot(2, 2, 2)
    max_velocities_data = [np.array([s.max_velocity for s in saccades])]
    distributions.bar_chart(ax=ax2, data=max_velocities_data,
                            labels=["All Saccades"], title="Maximum Velocities", xlabel="Max Velocity (pixels/ms)",
                            **kwargs)

    # amplitude distribution
    ax3 = fig.add_subplot(2, 2, 3)
    amplitude_data = [np.array([s.amplitude for s in saccades])]
    distributions.bar_chart(ax=ax3, data=amplitude_data,
                            labels=["All Saccades"], title="Amplitude", xlabel="Amplitude (°)", **kwargs)

    # azimuth distribution (polar)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    azimuth_data = [np.array([s.azimuth for s in saccades])]
    distributions.rose_chart(ax=ax4, data=azimuth_data, labels=["All Saccades"], title="Azimuth (°)", **kwargs)
    return fig

