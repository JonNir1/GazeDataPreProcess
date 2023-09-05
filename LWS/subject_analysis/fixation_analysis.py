import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import Config.experiment_config as cnfg
import Visualization.visualization_utils as visutils
import Visualization.dynamics as dynamics
import Visualization.distributions as distributions
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def split_by_target_proximity(fixations: List[LWSFixationEvent],
                              proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                              ignore_outliers: bool = True) -> Tuple[
        List[LWSFixationEvent], List[LWSFixationEvent], List[LWSFixationEvent]]:
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    target_proximal_fixations = [f for f in fixations if f.visual_angle_to_closest_target <= proximity_threshold]
    target_marking_fixations = [f for f in fixations if f.is_mark_target_attempt()]
    target_distal_fixations = [f for f in fixations if f.visual_angle_to_closest_target > proximity_threshold]
    if ignore_outliers:
        target_proximal_fixations = [f for f in target_proximal_fixations if not f.is_outlier]
        target_marking_fixations = [f for f in target_marking_fixations if not f.is_outlier]
        target_distal_fixations = [f for f in target_distal_fixations if not f.is_outlier]
    return target_proximal_fixations, target_marking_fixations, target_distal_fixations


def plot_fixation_comparison(fixation_groups: List[List[LWSFixationEvent]], group_names: List[str],
                             ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if len(fixation_groups) != len(group_names):
        raise ValueError(
            f"Number of groups ({len(fixation_groups)}) must match number of group names ({len(group_names)})")
    if ignore_outliers:
        fixation_groups = [[f for f in group if not f.is_outlier] for group in fixation_groups]

    title = "Fixation Feature Comparison"
    if "title" in kwargs:
        title = title + f"\n{kwargs.pop('title')}"
    fig = visutils.set_figure_properties(fig=None, title=title, figsize=kwargs.pop("figsize", (30, 24)),
                                         title_height=kwargs.pop("title_height", 0.93), **kwargs)

    # % outliers
    ax1 = fig.add_subplot(2, 3, 1)  # top left
    # TODO: create a bar chart with the % of outliers in each group

    # durations
    ax2 = fig.add_subplot(2, 3, 2)  # top middle
    durations_data = [np.array([f.duration for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax2, datasets=durations_data, data_labels=group_names,
                            title="Durations", xlabel="Duration (ms)", **kwargs)
    # dispersion
    ax4 = fig.add_subplot(2, 3, 4)  # bottom left
    dispersion_data = [np.array([f.dispersion for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax4, datasets=dispersion_data, data_labels=group_names,
                            title="Max Dispersion", xlabel="Max Dispersion (pixels)", **kwargs)
    # angle to target
    ax5 = fig.add_subplot(2, 3, 5)  # bottom middle
    distance_data = [np.array([f.visual_angle_to_closest_target for f in group]) for group in fixation_groups]
    distributions.bar_chart(ax=ax5, datasets=distance_data, data_labels=group_names,
                            title="Angle to Target", xlabel="Angle to Target (°)", **kwargs)

    # velocity dynamics & pupil size dynamics - in the right column with shared x-axis
    # we interpolate each timeseries to the maximum duration of all fixations in the group, normalize them to
    # range [0, 1] and plot them on the same axis
    ax6 = fig.add_subplot(2, 3, 6)  # bottom right
    ax3 = fig.add_subplot(2, 3, 3, sharex=ax6)  # top right

    # velocity dynamics
    velocity_data = [[f.get_velocity_series() for f in group] for group in fixation_groups]
    dynamics.dynamic_profile(ax=ax3, datasets=velocity_data, data_labels=group_names,
                             title="Velocity Dynamics", xlabel="Time (ms)", ylabel="Velocity (°/s)", **kwargs)
    # pupil size dynamics
    pupil_data = [[f.get_pupil_series() for f in group] for group in fixation_groups]
    dynamics.dynamic_profile(ax=ax6, datasets=pupil_data, data_labels=group_names,
                             title="Pupil Size Dynamics", xlabel="Time (ms)", ylabel="Pupil Size (mm)", **kwargs)
    return fig


def plot_feature_dynamics(fixation_groups: List[List[LWSFixationEvent]], group_names: List[str],
                          ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    """
    Creates a N×2 figure with the temporal dynamics of velocity (left column) and pupil size (right column) for
    each group of fixations (N is the number of groups).
    Each subplot shows the temporal dynamics of the given feature for a different group.

    :param fixation_groups: A list of lists of fixations. Each list of fixations represents a group.
    :param group_names: A list of names for each group.
    :param ignore_outliers: If True, outliers will be ignored.

    :return: A matplotlib Figure object.

    :raises ValueError: If the number of groups does not match the number of group names.
    """
    if len(fixation_groups) != len(group_names):
        raise ValueError(
            f"Number of groups ({len(fixation_groups)}) must match number of group names ({len(group_names)})")
    if ignore_outliers:
        fixation_groups = [[f for f in group if not f.is_outlier] for group in fixation_groups]
    if ("colors" not in kwargs) or (len(kwargs["colors"]) != len(fixation_groups)):
        colors = [plt.get_cmap("tab20")(2*i+1) for i in range(len(fixation_groups))]
    else:
        colors = kwargs.pop("colors")

    title = "Fixation Feature Dynamics"
    if "title" in kwargs:
        title = title + f"\n{kwargs.pop('title')}"
    fig = visutils.set_figure_properties(fig=None,
                                         title=title,
                                         figsize=kwargs.pop("figsize", (30, 24)),
                                         title_height=kwargs.pop("title_height", 0.93),
                                         **kwargs)

    n_groups = len(fixation_groups)
    axes = np.full((n_groups, 2), np.nan, dtype=object)
    for i in range(n_groups):
        i = n_groups - i - 1  # reverse order
        if i == n_groups - 1:
            axes[i, 0] = fig.add_subplot(n_groups, 2, 2 * i + 1)
            axes[i, 1] = fig.add_subplot(n_groups, 2, 2 * i + 2)
        else:
            # use same x-axis as plot at the bottom of the column
            axes[i, 0] = fig.add_subplot(n_groups, 2, 2 * i + 1, sharex=axes[n_groups-1, 0])
            axes[i, 1] = fig.add_subplot(n_groups, 2, 2 * i + 2, sharex=axes[n_groups-1, 1])

    for i, (group, name) in enumerate(zip(fixation_groups, group_names)):
        # velocities
        left_title = "Velocity Dynamics" if i == 0 else ""
        dynamics.velocity_profile(group, axes[i, 0], show_individual=False, show_peak=True,
                                  title=left_title, data_labels=[name], xlabel="", primary_color=colors[i], **kwargs)

        # pupils
        right_title = "Pupil Size Dynamics" if i == 0 else ""
        dynamics.pupil_size_profile(group, axes[i, 1], show_individual=False, show_peak=True,
                                    title=right_title, data_labels=[name], xlabel="", primary_color=colors[i], **kwargs)
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
