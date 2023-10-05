import numpy as np
import pandas as pd
from typing import List
from itertools import pairwise

import Config.experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.SubjectAnalysis.search_analysis.target_identification import get_target_identification_data

INSTANCES_DF_NAME = "lws_instances"
RATES_DF_BASE_NAME = "lws_rates"


def identify_lws_for_varying_thresholds(subject: LWSSubject,
                                        proximity_thresholds: np.ndarray,
                                        time_difference_thresholds: np.ndarray) -> pd.DataFrame:
    """
    For each (trial, proximity_threshold, time_difference_threshold) triplet, identifies the LWS instances in the
    trial. Returns a 3D dataframe where each cell contains a boolean array of the same length as the trial's fixations,
    where each element of the array is either True/False, depending on whether the corresponding gaze event is a
    LWS instance or not.

    The resulting DataFrame's is indexed by LWSTrial and the columns are a MultiIndex of the proximity-thresholds and
    time-difference thresholds.

    NOTE depending on the amount of varying thresholds, this may take 30-60 minutes to run for a single subject!
    """
    columns_multiindex = pd.MultiIndex.from_product([proximity_thresholds, time_difference_thresholds],
                                                    names=["proximity_threshold", "time_difference_threshold"])
    all_trials = subject.get_trials()
    is_lws_instance = pd.DataFrame(index=all_trials, columns=columns_multiindex)
    is_lws_instance.index.name = "trial"

    for trial in all_trials:
        for prox in proximity_thresholds:
            for td in time_difference_thresholds:
                is_lws_instance.loc[trial, (prox, td)] = _identify_lws_instances(trial,
                                                                                 proximity_threshold=prox,
                                                                                 time_difference_threshold=td)
    is_lws_instance.name = INSTANCES_DF_NAME
    return is_lws_instance


def calculate_lws_rates(subject: LWSSubject, proximal_fixations_only: bool) -> pd.DataFrame:
    """
    Calculates the LWS rate for all the subject's trials & for each (proximity_threshold, time_difference_threshold),
    based on the pre-computed `is_lws_instance` DataFrame that holds a boolean array for each triplet, indicating whether
    each fixation in the trial is a LWS instance (for those thresholds) or not.

    If the subject doesn't have a pre-computed `is_lws_instance` DataFrame, or if it's missing some of the (trial,
    proximity_threshold, time_difference_threshold) triplets, then the returned DataFrame will have NaN values for those
    triplets.
    """
    # attempt to load the data from disk
    df_path = subject.get_dataframe_path(INSTANCES_DF_NAME)
    try:
        is_lws_df = pd.read_pickle(df_path)
    except FileNotFoundError:
        is_lws_df = pd.DataFrame(np.nan,
                                 index=subject.get_trials(),
                                 columns=pd.MultiIndex.from_product(
                                     iterables=[[cnfg.THRESHOLD_VISUAL_ANGLE], [SaccadeEvent.MAX_DURATION]],
                                     names=["proximity_threshold", "time_difference_threshold"]))
        is_lws_df.index.name = "trial"

    rates_df = pd.DataFrame(np.nan, index=is_lws_df.index, columns=is_lws_df.columns)
    for trial in is_lws_df.index:
        for (prox, td) in is_lws_df.columns:
            # count the number of fixations in the trial:
            fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION)
            if proximal_fixations_only:
                fixations = list(filter(lambda f: f.visual_angle_to_closest_target <= prox, fixations))
            num_fixations = len(fixations)

            # calculate the LWS rate of this trial, for each (proximity_threshold, time_difference_threshold) pair:
            if num_fixations == 0:
                rates_df.loc[trial, (prox, td)] = np.nan
                continue
            is_lws_lst = is_lws_df.loc[trial, (prox, td)]
            if not is_lws_lst:
                rates_df.loc[trial, (prox, td)] = np.nan
                continue
            num_lws_instances = np.sum(is_lws_lst)
            rates_df.loc[trial, (prox, td)] = num_lws_instances / num_fixations

    rates_df.name = RATES_DF_BASE_NAME + ("_proximal_fixations" if proximal_fixations_only else "_all_fixations")
    return rates_df


def _identify_lws_instances(trial: LWSTrial,
                            proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                            time_difference_threshold: float = SaccadeEvent.MAX_DURATION) -> List[bool]:
    """
    Identifies the LWS instances in the given trial, and returns a list of the same length as the number of fixations
    in the trial, where each element is either:
        - True: if the corresponding fixation event is a LWS instance
        - False: if the corresponding fixation event is not a LWS instance

    See identification criteria in the docstring of `_check_lws_instance_standalone_criteria` and
    `_check_lws_instance_pairwise_criteria`.

    Note: this function assumes that the trial's gaze events are sorted by their start time.
    """
    target_info = get_target_identification_data(trial, proximity_threshold=proximity_threshold)
    fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION)

    # start with the last fixation
    is_lws_instance = [_check_lws_instance_standalone_criteria(fixations[-1], target_info, proximity_threshold)]

    # work our way backwards in pairs of fixations
    fixation_pairs = list(pairwise(fixations))
    for curr_fixation, next_fixation in fixation_pairs[::-1]:
        standalone_criterion = _check_lws_instance_standalone_criteria(curr_fixation,
                                                                       target_info,
                                                                       proximity_threshold)
        pairwise_criterion = _check_lws_instance_pairwise_criteria(curr_fixation,
                                                                   next_fixation,
                                                                   proximity_threshold=proximity_threshold,
                                                                   min_time_difference=time_difference_threshold,
                                                                   is_other_fixation_lws_instance=is_lws_instance[-1])
        is_lws_instance.append(standalone_criterion and pairwise_criterion)

    # reverse the list so that it's in the same order as the trial's fixations
    is_lws_instance = is_lws_instance[::-1]
    return is_lws_instance


def _check_lws_instance_standalone_criteria(fixation: LWSFixationEvent,
                                            target_identification_data: pd.DataFrame,
                                            proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE) -> bool:
    """
    Checks if the given fixation meets the standalone criteria required for a LWS instance: fixation needs to be close
    to a target that was not identified up until the end of the fixation.
        - If the fixation is not close to any target, then it does not meet the criteria.
        - If the fixation is close to a never-identified target, then it meets the criteria.
        - If the fixation is close to a target that was identified *after* the fixation ended, then it meets the criteria.
        - Otherwise it does not meet the criteria.
    """
    if fixation.end_time == fixation.trial.end_time:
        # the trial ended during this fixation, so we cannot say that the subject was unaware of the target at the end
        # of the fixation --> return False
        return False
    if fixation.visual_angle_to_closest_target > proximity_threshold:
        # fixation is not close to any target, so it cannot be a LWS fixation --> return False
        return False
    time_identified = target_identification_data.loc[fixation.closest_target_id, "time_identified"]
    if np.isnan(time_identified):
        # fixation is close to a never-identified target, so it may be a LWS fixation --> return True
        return True
    # if the fixation ended before the target was identified, then it could be a LWS fixation
    return fixation.end_time < time_identified


def _check_lws_instance_pairwise_criteria(curr_fixation: LWSFixationEvent,
                                          other_fixation: LWSFixationEvent,
                                          proximity_threshold: float,
                                          min_time_difference: float,
                                          is_other_fixation_lws_instance: bool) -> bool:
    """
    Checks if the given fixation meets the pairwise criteria required for a LWS instance:
        1- If the other fixation is in the target-helper region of the stimulus, then the current fixation doesn't meet
            the criteria.
        2- If the other fixation is closer to a different target, then the current fixation meets the criteria.
        3- If the other fixation is on the same target, but it's not close enough to the target, then the current
            fixation meets the criteria.
        4- If the other fixation is on the same target, but it started too long after the current fixation ended, then
            the current fixation meets the criteria.
        5- Otherwise the current fixation only meets the criteria if the other fixation is a LWS instance.
    """
    if other_fixation.is_in_rectangle(cnfg.STIMULUS_BOTTOM_STRIP_TOP_LEFT, cnfg.STIMULUS_BOTTOM_STRIP_BOTTOM_RIGHT):
        # next fixation is in the target-helper region of the stimulus, meaning that the subject (rightfully) suspects
        # that the current fixation is on a target --> the current fixation cannot be a LWS instance
        return False
    if other_fixation.closest_target_id != curr_fixation.closest_target_id:
        # next fixation is closer to a different target --> the current fixation could be a LWS instance
        return True
    if other_fixation.visual_angle_to_closest_target > proximity_threshold:
        # next fixation is on the same target, but it's not close enough to the target --> the current fixation could
        # be a LWS instance
        return True
    if other_fixation.start_time - curr_fixation.end_time > min_time_difference:
        # both fixations are on the same target, but the next fixation started too long after the current fixation
        # ended --> the current fixation could be a LWS instance
        return True

    # reached here if the subject uses both fixations to examine the same target, and they are close enough in time
    # for us to extrapolate from the next fixation onto the current one. so the current fixation is a LWS instance iff
    # the next fixation is also a LWS instance.
    return is_other_fixation_lws_instance
