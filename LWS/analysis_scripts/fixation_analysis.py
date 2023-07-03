import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import Config.experiment_config as cnfg
import Utils.oop_utils as oop
import Visualization.dynamics as dyn
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def dynamics_figure(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                               proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]
    proximal_fixations = [f for f in fixations if f.visual_angle_to_target <= proximity_threshold]
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt]

    fig, axes = plt.subplots(3, 2, figsize=kwargs.get("figsize", (20, 30)), sharex='col')
    fig.suptitle(kwargs.pop("title", f"Fixation Dynamics"), y=0.92, fontsize=kwargs.get("title_size", 16))

    # velocities
    ax = dyn.velocity_profile(fixations, axes[0, 0], show_individual=False, show_peak=True,
                              title="Velocity Dynamics", data_label="All Fixations", primary_color='darkblue', **kwargs)
    ax = dyn.velocity_profile(proximal_fixations, axes[1, 0], show_individual=False, show_peak=True,
                              title="", data_label="Proximal Fixations", primary_color='darkred', **kwargs)
    ax = dyn.velocity_profile(marking_fixations, axes[2, 0], show_individual=False, show_peak=True,
                              title="", data_label="Marking Fixations", primary_color='darkgreen', **kwargs)

    # pupils
    ax = dyn.pupil_size_profile(fixations, axes[0, 1], show_individual=False, show_peak=True,
                                title="Pupil Size Dynamics", data_label="All Fixations", primary_color='darkblue', **kwargs)
    ax = dyn.pupil_size_profile(proximal_fixations, axes[1, 1], show_individual=False, show_peak=True,
                                title="", data_label="Proximal Fixations", primary_color='darkred', **kwargs)
    ax = dyn.pupil_size_profile(marking_fixations, axes[2, 1], show_individual=False, show_peak=True,
                                title="", data_label="Marking Fixations", primary_color='darkgreen', **kwargs)
    return fig


def histograms_figure(fixations: List[LWSFixationEvent], ignore_outliers: bool = True,
                               proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Figure:
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig, axes = plt.subplots(2, 3, figsize=kwargs.get("figsize", (30, 15)))
    title = kwargs.get("title", f"Fixation Summary")
    fig.suptitle(title, y=0.98, fontsize=kwargs.get("title_size", 16))

    # durations
    ax = _compare_property_distributions(fixations, axes[0, 0], "duration", "ms", proximity_threshold, **kwargs)
    # max dispersion
    ax = _compare_property_distributions(fixations, axes[0, 1], "max_dispersion", "px", proximity_threshold, **kwargs)
    # angle to target
    ax = _compare_property_distributions(fixations, axes[0, 2], "visual_angle_to_target", "°", proximity_threshold, **kwargs)
    # max velocity
    ax = _compare_property_distributions(fixations, axes[1, 0], "max_velocity", "px/s", proximity_threshold, **kwargs)
    # mean velocity
    ax = _compare_property_distributions(fixations, axes[1, 1], "mean_velocity", "px/s", proximity_threshold, **kwargs)
    # mean pupil size
    ax = _compare_property_distributions(fixations, axes[1, 2], "mean_pupil_size", "mm", proximity_threshold, **kwargs)
    return fig


def _compare_property_distributions(fixations: List[LWSFixationEvent], ax: plt.Axes,
                                    property_name: str, property_units: str,
                                    proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE, **kwargs) -> plt.Axes:
    nbins = kwargs.get("nbins", 20)
    all_fixations_percentages, all_fixations_centers = __calculate_distribution(
        data=[oop.get_property(f, property_name) for f in fixations],
        nbins=nbins, min_threshold=1
    )
    proximal_fixations = [f for f in fixations if f.visual_angle_to_target <= proximity_threshold]
    proximal_fixations_percentages, proximal_fixations_centers = __calculate_distribution(
        data=[oop.get_property(f, property_name) for f in proximal_fixations],
        nbins=nbins, min_threshold=1
    )
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt]
    marking_fixations_percentages, marking_fixations_centers = __calculate_distribution(
        data=[oop.get_property(f, property_name) for f in marking_fixations],
        nbins=nbins, min_threshold=1
    )

    bar_width = np.min([np.min(np.diff(all_fixations_centers)) * 0.9,
                        np.min(np.diff(proximal_fixations_centers)) * 0.9,
                        np.min(np.diff(marking_fixations_centers)) * 0.9])
    face_colors = kwargs.get("face_colors", ["lightblue", "lightcoral", "lightgreen"])
    edge_colors = kwargs.get("edge_colors", ["darkblue", "darkred", "darkgreen"])
    ax.bar(all_fixations_centers, all_fixations_percentages, width=bar_width,
           facecolor=face_colors[0], edgecolor=edge_colors[0], align='center', alpha=0.9,
           label="All Fixations")
    ax.bar(proximal_fixations_centers, proximal_fixations_percentages, width=bar_width,
           facecolor=face_colors[1], edgecolor=edge_colors[1], align='center', alpha=0.8,
           label="Target-Proximal Fixations")
    ax.bar(marking_fixations_centers, marking_fixations_percentages, width=bar_width,
           facecolor=face_colors[2], edgecolor=edge_colors[2], align='center', alpha=0.7,
           label="Target-Marking Fixations")
    ax.set_title(" ".join([s.capitalize() for s in property_name.split("_")]), fontsize=kwargs.get("title_size", 14))
    text_size = kwargs.get("text_size", 12)
    ax.set_ylabel("%", fontsize=text_size)
    ax.set_xlabel(f"{property_units}", fontsize=text_size)
    ax.legend(loc=kwargs.get('legend_location', 'upper right'), fontsize=text_size)
    return ax


def __calculate_distribution(data: List[float], nbins: int, min_threshold: Optional[float]) -> (np.ndarray, np.ndarray):
    """
    Calculates the distribution of the given data, and returns the percentages and centers of the bins. Ignores bins
    with percentage less than min_threshold.
    """
    counts, edges = np.histogram(data, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    percentages = 100 * counts / np.sum(counts)
    assert len(percentages) == len(centers), "Percentages and centers must have same length"

    if min_threshold is not None:
        centers = centers[percentages >= min_threshold]
        percentages = percentages[percentages >= min_threshold]
    return percentages, centers
