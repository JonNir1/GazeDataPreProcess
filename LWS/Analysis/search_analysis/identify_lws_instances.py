import numpy as np
import pandas as pd
from typing import List, Union
from itertools import pairwise

import Config.experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.Analysis.search_analysis.target_identification import get_target_identification_data

DF_NAME = "lws_instances"


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

    NOTE this may take 30-60 minutes to run for a single subject!
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
    return is_lws_instance


def load_or_compute_lws_instances(trial: LWSTrial,
                                  proximity_threshold: float,
                                  time_difference_threshold: float) -> List[bool]:
    # attempt to load the data from disk
    df_path = trial.subject.get_dataframe_path(DF_NAME)
    try:
        is_lws_df = pd.read_pickle(df_path)
    except FileNotFoundError:
        is_lws_df = pd.DataFrame(index=trial.subject.get_trials(),
                          columns=pd.MultiIndex.from_product(
                              iterables=[[proximity_threshold], [time_difference_threshold]],
                              names=["proximity_threshold", "time_difference_threshold"]))
        is_lws_df.index.name = "trial"

    try:
        trial_is_lws = is_lws_df.loc[trial, (proximity_threshold, time_difference_threshold)]
    except KeyError:
        trial_is_lws = _identify_lws_instances(trial,
                                               proximity_threshold=proximity_threshold,
                                               time_difference_threshold=time_difference_threshold)
        is_lws_df.loc[trial, (proximity_threshold, time_difference_threshold)] = trial_is_lws
        is_lws_df.to_pickle(df_path)
    return trial_is_lws


def _identify_lws_instances(trial: LWSTrial,
                            proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                            time_difference_threshold: float = SaccadeEvent.MAX_DURATION) -> List[Union[bool, float]]:
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
