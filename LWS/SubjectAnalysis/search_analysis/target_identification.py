import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

import constants as cnst
import Config.experiment_config as cnfg
import Utils.array_utils as arr_utils
import Visualization.visualization_utils as visutils
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSArrayStimulus import LWSStimulusTypeEnum


def plot_identification_angle_distribution(subject: LWSSubject,
                                           max_angle_from_target: float = 2 * cnfg.THRESHOLD_VISUAL_ANGLE) -> plt.Figure:
    data_labels = ["all"] + [f"{stim_type}" for stim_type in LWSStimulusTypeEnum]

    # extract the angles of which each target was identified:
    all_identification_angles = pd.DataFrame.from_dict(
        {tr: get_target_identification_data(tr, max_angle_from_target)['distance_identified']
         for tr in subject.get_trials()},
        orient='index')
    identification_angles = {}
    for label, stim_type in zip(data_labels, [None] + list(LWSStimulusTypeEnum)):
        angles = all_identification_angles.loc[subject.get_trials(stim_type)].values.flatten()
        num_targets = angles.size - np.isnan(angles).sum()
        angles = angles[np.isfinite(angles)]  # remove unidentified targets
        angles = pd.Series(np.sort(angles), name=label)
        angles.index = np.arange(1, len(angles) + 1) / num_targets
        identification_angles[label] = angles

    fig, ax = plt.subplots(tight_layout=True)
    ax = visutils.generic_line_chart(ax,
                                     data_labels=[label for label in identification_angles.keys()],
                                     xs=[identification_angles[label].values for label in identification_angles.keys()],
                                     ys=[100 * identification_angles[label].index.values for label in
                                         identification_angles.keys()])
    ax = visutils.set_axes_properties(ax,
                                      xlabel="Identification Angle (°)",
                                      ylabel="% Targets Identified",
                                      show_legend=True)
    fig = visutils.set_figure_properties(fig=fig, figsize=(12, 8), tight_layout=True,
                                         title="Angle from Target When Identified\n(Cumulative Distribution)",
                                         title_height=0.98)
    return fig


def get_target_identification_data(trial: LWSTrial,
                                   max_angle_from_target: float = 2 * cnfg.THRESHOLD_VISUAL_ANGLE,
                                   identification_seq: np.ndarray = cnfg.TARGET_IDENTIFICATION_SEQUENCE) -> pd.DataFrame:
    """
    For each of the trial's targets, extracts the following information:
        - icon_path: full path to the icon file
        - icon_category: category of the icon (face, animal, etc.)
        - center_x: x coordinate of the icon center
        - center_y: y coordinate of the icon center
        - distance_identified: distance (in visual angle) between the target and the gaze when the target was
            identified by the subject
        - time_identified: time (in milliseconds) when the target was identified by the subject
        - time_confirmed: time (in milliseconds) when the target was confirmed by the subject

    :param trial: the trial to analyze
    :param max_angle_from_target: float - maximum visual angle (in deg) from target to count as identified
    :param identification_seq: np.ndarray - sequence of triggers indicating that a target was identified by the subject

    Returns a dataframe with shape (num_targets, 7), where each row corresponds to a target.
    """
    if max_angle_from_target <= 0:
        raise ValueError(f"Invalid `max_angle_from_target`: {max_angle_from_target}")
    if max_angle_from_target is None or np.isnan(max_angle_from_target):
        max_angle_from_target = np.inf

    # extract relevant columns from the behavioral data
    behavioral_data = trial.get_behavioral_data()
    columns = ([cnst.MICROSECONDS, cnst.TRIGGER, "closest_target"] +
               [col for col in behavioral_data.columns if col.startswith(f"{cnst.DISTANCE}_{cnst.TARGET}")])
    behavioral_df = pd.DataFrame(behavioral_data.get(columns), columns=columns)

    res = pd.DataFrame(np.full((trial.num_targets, 3), np.inf),
                       columns=["distance_identified", "time_identified", "time_confirmed"])
    for i in range(trial.num_targets):
        proximal_behavioral_df = behavioral_df[behavioral_df["closest_target"] == i]

        # check if target was ever identified by the subject
        identification_idxs = arr_utils.find_sequences_in_sparse_array(proximal_behavioral_df[cnst.TRIGGER].values,
                                                                       sequence=identification_seq)
        if len(identification_idxs) == 0:
            # this target was never identified
            continue

        # check if any of the target's identification attempts were from below the threshold distance
        identification_distances = np.array(
            [proximal_behavioral_df.iloc[first_idx][f"{cnst.DISTANCE}_{cnst.TARGET}{i}"]
             for first_idx, last_idx in identification_idxs])
        proximal_identifications = np.where(identification_distances < max_angle_from_target)[0]
        if len(proximal_identifications) == 0:
            # no proximal identification attempts
            continue

        # find the start & end idxs of the first identification attempt that was from below the threshold distance
        first_proximal_identification = min(proximal_identifications)
        first_proximal_identification_idxs = identification_idxs[first_proximal_identification]
        first_idx, last_idx = first_proximal_identification_idxs
        res.loc[i, "distance_identified"] = proximal_behavioral_df.iloc[first_idx][
            f"{cnst.DISTANCE}_{cnst.TARGET}{i}"]
        res.loc[i, "time_identified"] = proximal_behavioral_df.iloc[first_idx][
                                            cnst.MICROSECONDS] / cnst.MICROSECONDS_PER_MILLISECOND
        res.loc[i, "time_confirmed"] = proximal_behavioral_df.iloc[last_idx][
                                           cnst.MICROSECONDS] / cnst.MICROSECONDS_PER_MILLISECOND
    return pd.concat([trial.get_targets(), res], axis=1)


def _calc_identification_angle_histogram(subject: LWSSubject,
                                         max_angle_from_target: float = 2 * cnfg.THRESHOLD_VISUAL_ANGLE,
                                         stim_type: Optional[LWSStimulusTypeEnum] = None,
                                         nbins: int = 20) -> pd.Series:
    """
    Calculates the distribution of visual-angles from targets when they were identified by the subject.
    Unidentified targets are counted as angle = np.inf.

    :param subject: the subject to analyze
    :param max_angle_from_target: float - maximum visual angle (in deg) from target to count as identified
    :param stim_type: LWSStimulusTypeEnum - if given, only trials with this stimulus type will be analyzed
    :param nbins: int - number of bins to use for the identified targets' angles histogram (must be positive)

    :return: a pd.Series indexed by the centers of the hostogram bins and values are % of identified targets with
    identification angle within each bin (including bin `np.inf` for unidentified targets).
    """
    if nbins <= 0:
        raise ValueError(f"Invalid `nbins`: {nbins}")
    target_identification_angles = pd.DataFrame.from_dict(
        {tr: get_target_identification_data(tr, max_angle_from_target)['distance_identified']
         for tr in subject.get_trials(stim_type)},
        orient='index')

    # nan values indicate there was no such target in the trial
    num_targets = target_identification_angles.size - np.isnan(target_identification_angles).sum().sum()

    # inf values indicate the target was never identified
    percent_unidentified = 100 * np.isinf(target_identification_angles).sum().sum() / num_targets

    # finite values indicate the target was identified when gaze was at angle < `max_angle_from_target`
    identification_angles = target_identification_angles.values.flatten()
    identification_angles = identification_angles[np.isfinite(identification_angles)]  # remove unidentified targets
    ident_percentages, ident_centers = arr_utils.calculate_distribution(data=identification_angles, nbins=nbins)

    # normalize percentages to account for unidentified targets
    ident_percentages = ident_percentages * len(identification_angles) / num_targets
    ident_distribution = pd.Series(ident_percentages, index=ident_centers)
    ident_distribution[np.inf] = percent_unidentified  # add percentage of unidentified targets
    return ident_distribution
