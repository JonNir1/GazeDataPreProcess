import os
import numpy as np
import pandas as pd
from typing import List

import Config.experiment_config as cnfg
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
from LWS.SubjectAnalysis.search_analysis.target_identification import get_target_identification_data

BASE_DF_NAME = 'return_to_roi'


def count_fixations_between_roi_visits_for_varying_thresholds(subject: LWSSubject,
                                                              proximity_thresholds: np.ndarray,
                                                              is_targets_rect_part_of_roi: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame of shape (num_trials, num_thresholds) where each cell contains an 2D numpy array of shape
    (num_targets, num_fixations) of the specific trial. Each value in the array indicates if the current fixation is
    outside the specific target's RoI (NaN value), or - if the fixation is inside the RoI - the number of fixations that
    occurred between the current fixation and the next time that the subject had a fixation inside this RoI (or np.inf
    if the RoI was never revisited).
    """
    all_trials = subject.get_trials()
    return_to_roi_counts = pd.DataFrame(index=subject.get_trials(), columns=proximity_thresholds)
    return_to_roi_counts.index.name = "trial"
    for trial in all_trials:
        for prox_thresh in proximity_thresholds:
            counts = count_fixations_between_roi_visits(trial, proximity_threshold=prox_thresh,
                                                        is_targets_rect_part_of_roi=is_targets_rect_part_of_roi)
            return_to_roi_counts.loc[trial, prox_thresh] = counts
    return return_to_roi_counts


def count_fixations_between_roi_visits(trial: LWSTrial,
                                       proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                                       is_targets_rect_part_of_roi: bool = False) -> np.ndarray:
    """
    Returns an array of shape (num_targets, num_fixations) where each value indicates is NaN if the current fixation is
    outside the specific target's RoI. If the fixation is inside the RoI, the value indicates the number of fixations
    that occurred between the current fixation and the next time that the subject had a fixation inside this RoI (or
    np.inf if the RoI was never revisited).
    """
    targets_info = get_target_identification_data(trial, proximity_threshold=proximity_threshold)
    fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION)

    fixation_counts = np.full((targets_info.shape[0], len(fixations)), np.nan)
    for i, target in targets_info.iterrows():
        is_in_roi = _check_if_in_roi(fixations, target, proximity_threshold, is_targets_rect_part_of_roi)
        counts = _count_false_between_true(is_in_roi)
        fixation_counts[i] = counts
    return fixation_counts


def _check_if_in_roi(fixations: List[LWSFixationEvent],
                     target_data: pd.Series,
                     proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                     is_targets_rect_part_of_roi: bool = False):
    """
    Returns a boolean array with the same length as `fixations`, where each value indicates whether the corresponding
    fixation is in the RoI (True) or not (False).

    :param fixations: list of LWSFixationEvent objects
    :param target_data: pd.Series containing the x-y coordinates of the target's center of mass (as well as other info)
    :param proximity_threshold: float indicating the maximum distance (in visual angle) from the target's center of mass
        to be considered "inside the RoI"
    :param is_targets_rect_part_of_roi: if True, the bottom strip of the screen (where the targets are presented) will
        be considered part of the RoI

    :return: `is_in_roi` - a boolean array indicating whether each fixation is inside the RoI
    """
    target_x, target_y = target_data['center_x'], target_data['center_y']
    fixations = pd.Series(fixations)
    is_in_roi = fixations.apply(
        lambda fix: fix.is_close_to_pixel(pixel=(target_x, target_y), threshold=proximity_threshold,
                                          threshold_units='deg'))
    if is_targets_rect_part_of_roi:
        is_in_targets_rect = fixations.apply(
            lambda fix: fix.is_in_rectangle(cnfg.STIMULUS_BOTTOM_STRIP_TOP_LEFT,
                                            cnfg.STIMULUS_BOTTOM_STRIP_BOTTOM_RIGHT))
        is_in_roi |= is_in_targets_rect
    return is_in_roi


def _count_false_between_true(bool_arr):
    res = np.full_like(bool_arr, np.nan, dtype=float)
    counts = np.inf  # if the RoI was never revisited, there are infinite fixations until revisit
    for i, val in enumerate(reversed(bool_arr)):
        if not val:
            counts += 1
            continue
        res[i] = counts
        counts = 0
    return np.flip(res)
