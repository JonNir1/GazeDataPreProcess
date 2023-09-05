import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import Config.experiment_config as cnfg
import Visualization.visualization_utils as visutils
import Visualization.dynamics as dynamics
import Visualization.distributions as distributions
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def split_by_target_proximity(fixations: List[LWSFixationEvent],
                              proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE) -> Tuple[
    List[LWSFixationEvent], List[LWSFixationEvent], List[LWSFixationEvent]]:
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    target_proximal_fixations = [f for f in fixations if f.visual_angle_to_closest_target <= proximity_threshold]
    target_marking_fixations = [f for f in fixations if f.is_mark_target_attempt()]
    target_distal_fixations = [f for f in fixations if f.visual_angle_to_closest_target > proximity_threshold]
    return target_proximal_fixations, target_marking_fixations, target_distal_fixations


def target_proximal_comparison(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                               proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt()]
    non_marking_proximal_fixations = [f for f in fixations if
                                      f.visual_angle_to_closest_target <= proximity_threshold and not f.is_mark_target_attempt()]
    data_labels = ["Marking Fixations", "Non-Marking Proximal Fixations"]
    fig = visutils.set_figure_properties(fig=None,
                                         title=kwargs.pop("title", f"Comparison of Target-Proximal Fixations"),
                                         figsize=kwargs.pop("figsize", (30, 24)),
                                         title_height=kwargs.pop("title_height", 0.93),
                                         **kwargs)
    # % outliers
    ax1 = fig.add_subplot(2, 3, 1)  # top left
    # TODO: create a stacked bar chart with the % of outliers in each group

    # durations
    ax2 = fig.add_subplot(2, 3, 2)  # top middle
    durations_data = [np.array([f.duration for f in marking_fixations]),
                      np.array([f.duration for f in non_marking_proximal_fixations])]
    distributions.bar_chart(ax=ax2, datasets=durations_data, data_labels=data_labels,
                            title="Durations", xlabel="Duration (ms)", **kwargs)
    # dispersion
    ax4 = fig.add_subplot(2, 3, 4)  # bottom left
    dispersion_data = [np.array([f.dispersion for f in marking_fixations]),
                       np.array([f.dispersion for f in non_marking_proximal_fixations])]
    distributions.bar_chart(ax=ax4, datasets=dispersion_data, data_labels=data_labels,
                            title="Max Dispersion", xlabel="Max Dispersion (pixels)", **kwargs)
    # angle to target
    ax5 = fig.add_subplot(2, 3, 5)  # bottom middle
    distance_data = [np.array([f.visual_angle_to_closest_target for f in marking_fixations]),
                     np.array([f.visual_angle_to_closest_target for f in non_marking_proximal_fixations])]
    distributions.bar_chart(ax=ax5, datasets=distance_data, data_labels=data_labels,
                            title="Angle to Target", xlabel="Angle to Target (°)", **kwargs)

    # velocity dynamics & pupil size dynamics - in the right column with shared x-axis
    # we interpolate each timeseries to the maximum duration of all fixations in the group, normalize them to
    # range [0, 1] and plot them on the same axis
    ax6 = fig.add_subplot(2, 3, 6)  # bottom right
    ax3 = fig.add_subplot(2, 3, 3, sharex=ax6)  # top right

    # velocity dynamics
    velocity_data = [[f.get_velocity_series() for f in marking_fixations],
                     [f.get_velocity_series() for f in non_marking_proximal_fixations]]
    dynamics.dynamic_profile(ax=ax3, datasets=velocity_data, data_labels=data_labels,
                             title="Velocity Dynamics", xlabel="Time (ms)", ylabel="Velocity (°/s)", **kwargs)
    # pupil size dynamics
    pupil_data = [[f.get_pupil_series() for f in marking_fixations],
                  [f.get_pupil_series() for f in non_marking_proximal_fixations]]
    dynamics.dynamic_profile(ax=ax6, datasets=pupil_data, data_labels=data_labels,
                             title="Pupil Size Dynamics", xlabel="Time (ms)", ylabel="Pupil Size (mm)", **kwargs)
    return fig


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
    proximal_fixations = [f for f in fixations if f.visual_angle_to_closest_target <= proximity_threshold]
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt()]
    fig = visutils.set_figure_properties(fig=None,
                                         title=kwargs.pop("title", f"Fixation Dynamics"),
                                         figsize=kwargs.pop("figsize", (30, 24)),
                                         title_height=kwargs.pop("title_height", 0.93),
                                         **kwargs)
    # velocities
    ax5 = fig.add_subplot(3, 2, 5)
    ax3 = fig.add_subplot(3, 2, 3, sharex=ax5)  # use same x-axis as plot at the bottom of the column
    ax1 = fig.add_subplot(3, 2, 1, sharex=ax5)  # use same x-axis as plot at the bottom of the column
    dynamics.velocity_profile(fixations, ax1, show_individual=False, show_peak=True,
                              title="Velocity Dynamics", data_labels=["All Fixations"], xlabel="",
                              primary_color='darkblue', **kwargs)
    dynamics.velocity_profile(proximal_fixations, ax3, show_individual=False, show_peak=True,
                              title="", data_labels=["Proximal Fixations"], xlabel="", primary_color='darkred', **kwargs)
    dynamics.velocity_profile(marking_fixations, ax5, show_individual=False, show_peak=True,
                              title="", data_labels=["Marking Fixations"], primary_color='darkgreen', **kwargs)

    # pupils
    ax6 = fig.add_subplot(3, 2, 6)
    ax4 = fig.add_subplot(3, 2, 4, sharex=ax6)  # use same x-axis as plot at the bottom of the column
    ax2 = fig.add_subplot(3, 2, 2, sharex=ax6)  # use same x-axis as plot at the bottom of the column
    dynamics.pupil_size_profile(fixations, ax2, show_individual=False, show_peak=True,
                                title="Pupil Size Dynamics", data_labels=["All Fixations"], xlabel="",
                                primary_color='darkblue', **kwargs)
    dynamics.pupil_size_profile(proximal_fixations, ax4, show_individual=False, show_peak=True,
                                title="", data_labels=["Proximal Fixations"], xlabel="", primary_color='darkred', **kwargs)
    dynamics.pupil_size_profile(marking_fixations, ax6, show_individual=False, show_peak=True,
                                title="", data_labels=["Marking Fixations"], primary_color='darkgreen', **kwargs)
    return fig


def plot_feature_distributions(fixation_groups: List[List[LWSFixationEvent]], group_names: List[str],
                               ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    """
    Creates a 2×3 figure with distributions of the following properties: fixation durations, max dispersion,
    angle to target, max velocity, mean velocity, and mean pupil size.
    Each subplot shows the distribution of the given feature for all groups.

    :param fixation_groups: A list of lists of fixations. Each list of fixations represents a group.
    :param group_names: A list of names for each group.
    :param ignore_outliers: If True, outliers will be ignored.

    :return: A matplotlib Figure object.

    :raises ValueError: If the number of groups does not match the number of group names.
    """
    if len(fixation_groups) != len(group_names):
        raise ValueError(f"Number of groups ({len(fixation_groups)}) must match number of group names ({len(group_names)})")
    if ignore_outliers:
        fixation_groups = [[f for f in group if not f.is_outlier] for group in fixation_groups]

    title = "Fixation Feature Distributions"
    if "title" in kwargs:
        title = title + f"\n{kwargs.pop('title')}"
    fig = visutils.set_figure_properties(fig=None, title=title, figsize=kwargs.pop("figsize", (30, 15)), **kwargs)

    # durations
    ax1 = fig.add_subplot(2, 3, 1)
    durations_data = [np.array([f.duration for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax1, datasets=durations_data, data_labels=group_names,
                            xlabel="Duration (ms)", title="Duration Distribution", **kwargs)
    # max dispersion
    ax2 = fig.add_subplot(2, 3, 2)
    max_dispersion_data = [np.array([f.dispersion for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax2, datasets=max_dispersion_data, data_labels=group_names,
                            title="Max Dispersion (px)", **kwargs)
    # angle to target
    ax3 = fig.add_subplot(2, 3, 3)
    angle_to_target_data = [np.array([f.visual_angle_to_closest_target for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax3, datasets=angle_to_target_data, data_labels=group_names,
                            title="Angle to Target (°)", **kwargs)
    # max velocity
    ax4 = fig.add_subplot(2, 3, 4)
    max_velocity_data = [np.array([f.max_velocity for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax4, datasets=max_velocity_data, data_labels=group_names,
                            title="Max Velocity (px/s)", **kwargs)
    # mean velocity
    ax5 = fig.add_subplot(2, 3, 5)
    mean_velocity_data = [np.array([f.mean_velocity for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax5, datasets=mean_velocity_data, data_labels=group_names,
                            title="Mean Velocity (px/s)", **kwargs)
    # mean pupil size
    ax6 = fig.add_subplot(2, 3, 6)
    mean_pupil_size_data = [np.array([f.mean_pupil_size for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax6, datasets=mean_pupil_size_data, data_labels=group_names,
                            title="Mean Pupil Size (mm)", **kwargs)
    return fig
