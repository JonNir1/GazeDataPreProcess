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
from LWS.Analysis.search_analysis.target_identification import target_identification_data


DF_NAME = "lws_instances"


def identify_lws_for_varying_thresholds(subject: LWSSubject,
                                        proximity_thresholds: np.ndarray,
                                        time_difference_thresholds: np.ndarray) -> pd.DataFrame:
    """
    For each (trial, proximity_threshold, time_difference_threshold) triplet, identifies the LWS instances in the
    trial. Returns a 3D dataframe where each cell contains a boolean array of the same length as the trial's gaze
    events, where each element of the array is either True/False/np.nan, depending on whether the corresponding gaze
    event is a LWS instance or not.

    The resulting DataFrame's is indexed by LWSTrial and the columns are a MultiIndex of the proximity-thresholds and
    time-difference thresholds.

    NOTE this may take 30-60 minutes to run for a single subject!
    """
    columns_multiindex = pd.MultiIndex.from_product([proximity_thresholds, time_difference_thresholds],
                                                    names=["proximity_threshold", "time_difference_threshold"])
    all_trials = subject.get_all_trials()
    is_lws_instance = pd.DataFrame(index=all_trials, columns=columns_multiindex)
    is_lws_instance.index.name = "trial"

    for trial in all_trials:
        for prox in proximity_thresholds:
            for td in time_difference_thresholds:
                is_lws_instance.loc[trial, (prox, td)] = _identify_lws_instances(trial,
                                                                                 proximity_threshold=prox,
                                                                                 time_difference_threshold=td)
    return is_lws_instance


def load_or_identify_lws_instances(trial: LWSTrial,
                                   proximity_threshold,
                                   time_difference_threshold) -> List[Union[bool, float]]:
    """
    To avoid re-identifying LWS instances for the same trial and threshold values, we save the results in a dataframe
    and load them if they exist. Otherwise, we identify the LWS instances and save them in the dataframe.
    """
    df_path = trial.subject.get_dataframe_path(DF_NAME)
    try:
        df = pd.read_pickle(df_path)
        if (proximity_threshold, time_difference_threshold) in df.columns:
            return df.loc[trial, (proximity_threshold, time_difference_threshold)]
    except FileNotFoundError:
        df = pd.DataFrame(index=trial.subject.get_all_trials(),
                          columns=pd.MultiIndex.from_product(iterables=[[proximity_threshold], [time_difference_threshold]],
                                                             names=["proximity_threshold", "time_difference_threshold"]))
        df.index.name = "trial"

    is_lws_instance = _identify_lws_instances(trial, proximity_threshold=proximity_threshold,
                                              time_difference_threshold=time_difference_threshold)
    df.loc[[trial], (proximity_threshold, time_difference_threshold)] = pd.Series([is_lws_instance], index=[trial])
    df.to_pickle(df_path)
    return is_lws_instance


def calculate_lws_rate(trial: LWSTrial,
                       proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                       time_difference_threshold: float = SaccadeEvent.MAX_DURATION,
                       proximal_fixations_only: bool = False) -> float:
    """
    Calculates the LWS rate for the given trial, which is the fraction of fixations that are LWS instances out of
    (a) all fixations in the trial; or (b) only the proximal fixations in the trial, depending on the value of the flag
    `proximal_fixations_only`.
    """
    is_lws_instance = load_or_identify_lws_instances(trial,
                                                     proximity_threshold=proximity_threshold,
                                                     time_difference_threshold=time_difference_threshold)
    num_lws_instances = np.nansum(is_lws_instance)
    fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION)
    if proximal_fixations_only:
        fixations = list(filter(lambda f: f.visual_angle_to_closest_target <= proximity_threshold, fixations))
    num_fixations = len(fixations)
    if num_fixations > 0:
        return num_lws_instances / num_fixations
    if num_lws_instances == 0 and num_fixations == 0:
        return np.nan
    raise ZeroDivisionError(f"num_lws_instances = {num_lws_instances},\tnum_fixations = {num_fixations}")


def _identify_lws_instances(trial: LWSTrial,
                            proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                            time_difference_threshold: float = SaccadeEvent.MAX_DURATION) -> List[Union[bool, float]]:
    """
    Identifies the LWS instances in the given trial, and returns a list of the same length as the trial's gaze events,
    where each element is either:
        - True: if the corresponding fixation event is a LWS instance
        - False: if the corresponding fixation event is not a LWS instance
        - np.nan: if the corresponding gaze event is not a fixation

    See identification criteria in the docstring of `_check_lws_instance_standalone_criteria` and
    `_check_lws_instance_pairwise_criteria`.

    Note: this function assumes that the trial's gaze events are sorted by their start time.
    """
    target_info = target_identification_data(trial, proximity_threshold=proximity_threshold)
    events = trial.get_gaze_events()
    fixation_idxs = np.where([e.event_type() == GazeEventTypeEnum.FIXATION for e in events])[0]
    is_lws_instance = np.full_like(events, np.nan)

    # start with the last fixation
    last_fixation_idx = fixation_idxs[-1]
    is_lws_instance[last_fixation_idx] = _check_lws_instance_standalone_criteria(events[last_fixation_idx], target_info,
                                                                                 proximity_threshold)

    # work our way backwards in pairs of fixations
    fixation_pair_idxs = list(pairwise(fixation_idxs))
    for curr_fixation_idx, next_fixation_idx in fixation_pair_idxs[::-1]:
        curr_fixation = events[curr_fixation_idx]
        next_fixation = events[next_fixation_idx]
        is_next_lws_instance = is_lws_instance[next_fixation_idx]
        standalone_criterion = _check_lws_instance_standalone_criteria(curr_fixation,
                                                                       target_info,
                                                                       proximity_threshold)
        pairwise_criterion = _check_lws_instance_pairwise_criteria(curr_fixation,
                                                                   next_fixation,
                                                                   proximity_threshold=proximity_threshold,
                                                                   min_time_difference=time_difference_threshold,
                                                                   is_other_fixation_lws_instance=is_next_lws_instance)
        is_lws_instance[curr_fixation_idx] = standalone_criterion and pairwise_criterion
    return list(is_lws_instance)


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
