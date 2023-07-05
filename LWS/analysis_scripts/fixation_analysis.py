import numpy as np
import matplotlib.pyplot as plt
from typing import List

import Config.experiment_config as cnfg
import Visualization.visualization_utils as visutils
import Visualization.dynamics as dynamics
import Visualization.distributions as distributions
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def target_proximal_comparison(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                               proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt]
    non_marking_proximal_fixations = [f for f in fixations if
                                      f.visual_angle_to_target <= proximity_threshold and not f.is_mark_target_attempt]
    fig = visutils.set_figure_properties(fig=None,
                                         title=kwargs.pop("title", f"Comparison of Target-Proximal Fixations"),
                                         figsize=kwargs.pop("figsize", (27, 21)), **kwargs)
    # durations
    # dispersion
    # distance from target
    # % outliers
    # velocity dynamics - overlayed
    # pupil size dynamics - overlayed
    return None


def dynamics_figure(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                    proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    """
    Creates a 3×2 figure depicting the temporal dynamics of velocity (left column) and pupil size (right column) for
    all fixations (top), target-proximal fixations (middle), and marking fixations (bottom).

    target-proximal fixations are defined as fixations with a visual angle to target less than or equal to the
    proximity threshold (default: 1.5°). target-marking fixations are defined as fixations during which the subject
    attempted to mark the target (i.e., the target-marking triggers were recorded).
    """
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    proximal_fixations = [f for f in fixations if f.visual_angle_to_target <= proximity_threshold]
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt]
    fig = visutils.set_figure_properties(fig=None, title=kwargs.pop("title", f"Fixation Dynamics"),
                                         figsize=kwargs.pop("figsize", (27, 21)), **kwargs)
    # velocities
    ax5 = fig.add_subplot(3, 2, 5)
    ax3 = fig.add_subplot(3, 2, 3, sharex=ax5)  # use same x-axis as plot at the bottom of the column
    ax1 = fig.add_subplot(3, 2, 1, sharex=ax5)  # use same x-axis as plot at the bottom of the column
    dynamics.velocity_profile(fixations, ax1, show_individual=False, show_peak=True,
                              title="Velocity Dynamics", data_label=["All Fixations"], xlabel="",
                              primary_color='darkblue', **kwargs)
    dynamics.velocity_profile(proximal_fixations, ax3, show_individual=False, show_peak=True,
                              title="", data_label=["Proximal Fixations"], xlabel="", primary_color='darkred', **kwargs)
    dynamics.velocity_profile(marking_fixations, ax5, show_individual=False, show_peak=True,
                              title="", data_labels=["Marking Fixations"], primary_color='darkgreen', **kwargs)

    # pupils
    ax6 = fig.add_subplot(3, 2, 6)
    ax4 = fig.add_subplot(3, 2, 4, sharex=ax6)  # use same x-axis as plot at the bottom of the column
    ax2 = fig.add_subplot(3, 2, 2, sharex=ax6)  # use same x-axis as plot at the bottom of the column
    dynamics.pupil_size_profile(fixations, ax2, show_individual=False, show_peak=True,
                                title="Pupil Size Dynamics", data_label=["All Fixations"], xlabel="",
                                primary_color='darkblue', **kwargs)
    dynamics.pupil_size_profile(proximal_fixations, ax4, show_individual=False, show_peak=True,
                                title="", data_label=["Proximal Fixations"], xlabel="", primary_color='darkred', **kwargs)
    dynamics.pupil_size_profile(marking_fixations, ax6, show_individual=False, show_peak=True,
                                title="", data_label=["Marking Fixations"], primary_color='darkgreen', **kwargs)
    return fig


def distributions_figure(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                         proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    """
    Creates a 2×3 figure with distributions of the following properties: fixation durations, max dispersion,
    angle to target, max velocity, mean velocity, and mean pupil size. Each histogram shows the distribution of
    all fixations (blue), target-proximal fixations (red), and target-marking fixations (green).

    target-proximal fixations are defined as fixations with a visual angle to target less than or equal to the
    proximity threshold (default: 1.5°). target-marking fixations are defined as fixations during which the subject
    attempted to mark the target (i.e., the target-marking triggers were recorded).
    """
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    target_proximal_fixations = [f for f in fixations if f.visual_angle_to_target <= proximity_threshold]
    target_marking_fixations = [f for f in fixations if f.is_mark_target_attempt]
    data_labels = ["All", "Target-Proximal", "Target-Marking"]
    fig = visutils.set_figure_properties(fig=None, title=kwargs.pop("title", f"Fixation Summary"),
                                         figsize=kwargs.pop("figsize", (30, 15)), **kwargs)
    # durations
    ax1 = fig.add_subplot(2, 3, 1)
    durations_data = [np.array([f.duration for f in fixations]),
                      np.array([f.duration for f in target_proximal_fixations]),
                      np.array([f.duration for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax1, data=durations_data, data_labels=data_labels,
                            title="Durations", xlabel="Duration (ms)", **kwargs)
    # max dispersion
    ax2 = fig.add_subplot(2, 3, 2)
    max_dispersion_data = [np.array([f.max_dispersion for f in fixations]),
                           np.array([f.max_dispersion for f in target_proximal_fixations]),
                           np.array([f.max_dispersion for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax2, data=max_dispersion_data, data_labels=data_labels,
                            title="Max Dispersion", xlabel="Max Dispersion (px)", **kwargs)
    # angle to target
    ax3 = fig.add_subplot(2, 3, 3)
    angle_to_target_data = [np.array([f.visual_angle_to_target for f in fixations]),
                            np.array([f.visual_angle_to_target for f in target_proximal_fixations]),
                            np.array([f.visual_angle_to_target for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax3, data=angle_to_target_data, data_labels=data_labels,
                            title="Angle to Target", xlabel="Angle to Target (°)", **kwargs)
    # max velocity
    ax4 = fig.add_subplot(2, 3, 4)
    max_velocity_data = [np.array([f.max_velocity for f in fixations]),
                         np.array([f.max_velocity for f in target_proximal_fixations]),
                         np.array([f.max_velocity for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax4, data=max_velocity_data, data_labels=data_labels,
                            title="Max Velocity", xlabel="Max Velocity (px/s)", **kwargs)
    # mean velocity
    ax5 = fig.add_subplot(2, 3, 5)
    mean_velocity_data = [np.array([f.mean_velocity for f in fixations]),
                          np.array([f.mean_velocity for f in target_proximal_fixations]),
                          np.array([f.mean_velocity for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax5, data=mean_velocity_data, data_labels=data_labels,
                            title="Mean Velocity", xlabel="Mean Velocity (px/s)", **kwargs)
    # mean pupil size
    ax6 = fig.add_subplot(2, 3, 6)
    mean_pupil_size_data = [np.array([f.mean_pupil_size for f in fixations]),
                            np.array([f.mean_pupil_size for f in target_proximal_fixations]),
                            np.array([f.mean_pupil_size for f in target_marking_fixations])]
    distributions.bar_chart(ax=ax6, data=mean_pupil_size_data, data_labels=data_labels,
                            title="Mean Pupil Size", xlabel="Mean Pupil Size (mm)", **kwargs)
    return fig
