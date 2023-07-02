import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import Utils.oop_utils as oop
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def fixation_histograms_figure(fixations: List[LWSFixationEvent], ignore_outliers: bool = True, **kwargs) -> plt.Figure:
    if ignore_outliers:
        fixations = [f for f in fixations if not f.is_outlier]

    fig, axes = plt.subplots(2, 3, figsize=kwargs.get("figsize", (30, 15)))
    fig.suptitle("Fixation Summary", y=0.98, fontsize=kwargs.get("title_size", 16))

    # durations
    ax = _compare_property_distributions(fixations, axes[0, 0], "duration", "ms", **kwargs)
    # max dispersion
    ax = _compare_property_distributions(fixations, axes[0, 1], "max_dispersion", "px", **kwargs)
    # mean pupil size
    ax = _compare_property_distributions(fixations, axes[0, 2], "mean_pupil_size", "mm", **kwargs)
    # max velocity
    ax = _compare_property_distributions(fixations, axes[1, 0], "max_velocity", "px/s", **kwargs)
    # mean velocity
    ax = _compare_property_distributions(fixations, axes[1, 1], "mean_velocity", "px/s", **kwargs)
    # angle to target
    ax = _compare_property_distributions(fixations, axes[1, 2], "visual_angle_to_target", "°", **kwargs)
    return fig


def _compare_property_distributions(fixations: List[LWSFixationEvent], ax: plt.Axes,
                                    property_name: str, property_units: str, **kwargs) -> plt.Axes:
    nbins = kwargs.get("nbins", 20)
    all_fixations_percentages, all_fixations_centers = __calculate_distribution(
        data=[oop.get_property(f, property_name) for f in fixations],
        nbins=nbins, min_threshold=1
    )
    marking_fixations = [f for f in fixations if f.is_mark_target_attempt]
    marking_fixations_percentages, marking_fixations_centers = __calculate_distribution(
        data=[oop.get_property(f, property_name) for f in marking_fixations],
        nbins=nbins, min_threshold=1
    )

    bar_width = np.min([np.min(np.diff(all_fixations_centers)) * 0.9, np.min(np.diff(marking_fixations_centers)) * 0.9])
    face_colors = kwargs.get("face_colors", ["lightblue", "lightgreen"])
    edge_colors = kwargs.get("edge_colors", ["darkblue", "darkgreen"])
    ax.bar(all_fixations_centers, all_fixations_percentages, width=bar_width,
           facecolor=face_colors[0], edgecolor=edge_colors[0], align='center', alpha=0.75, label="All Fixations")
    ax.bar(marking_fixations_centers, marking_fixations_percentages, width=bar_width,
           facecolor=face_colors[1], edgecolor=edge_colors[1], align='center', alpha=0.75, label="Target-Marking Fixations")
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
