# LWS PreProcessing Pipeline

import numpy as np
from typing import List, Tuple

import constants as cnst
import experiment_config as cnfg
import Utils.array_utils as au
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent


def gen_all_lws_events(trial: LWSTrial, screen_monitor: ScreenMonitor, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Generates all gaze events for the given trial and returns the events in a list sorted by their start time.
    """
    blink_events = gen_lws_gaze_events(cnst.BLINK, trial, screen_monitor)
    saccade_events = gen_lws_gaze_events(cnst.SACCADE, trial, screen_monitor)
    fixation_events = gen_lws_gaze_events(cnst.FIXATION, trial, screen_monitor)
    all_events = blink_events + saccade_events + fixation_events
    if drop_outliers:
        all_events = [event for event in all_events if not event.is_outlier]
    all_events.sort(key=lambda event: event.start_time)
    return all_events


def gen_lws_gaze_events(event_type: str, trial: LWSTrial, screen_monitor: ScreenMonitor) -> List[BaseGazeEvent]:
    """
    Identifies all chunks of data that belong to this type of event within the trial and creates a GazeEvent object
    for each chunk and returns the list of events.
    For more information see the generic implementation in GazeEvents.pre_processing_scripts.gen_gaze_events.py

    :param event_type: type of event to extract. Must be one of 'blink', 'saccade' or 'fixation'
    :param trial: LWSTrial object
    :param screen_monitor: ScreenMonitor object

    :return: list of GazeEvent objects

    :raises: ValueError: if `event_type` is not one of 'blink', 'saccade' or 'fixation'
    """
    event_type = event_type.lower()
    timestamps, x, y, is_event = __extract_raw_event_arrays(trial=trial, event_type=event_type)

    if event_type == cnst.BLINK:
        # use generic gaze events
        from GazeEvents.scripts.gen_gaze_events import gen_gaze_events
        blinks_list = gen_gaze_events(timestamps=timestamps, is_event=is_event, event_type=cnst.BLINK)
        return blinks_list

    separate_event_idxs = au.get_different_event_indices(is_event, min_length=cnfg.DEFAULT_MINIMUM_SAMPLES_PER_EVENT)

    if event_type == cnst.SACCADE:
        # create LWSSaccadeEvents
        from LWS.DataModels.LWSSaccadeEvent import LWSSaccadeEvent
        distance = trial.get_subject_info().distance_to_screen
        saccades_list = []
        for idxs in separate_event_idxs:
            sacc = LWSSaccadeEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs],
                                   distance=distance, pixel_size=screen_monitor.pixel_size)
            saccades_list.append(sacc)
        return saccades_list

    if event_type == cnst.FIXATION:
        # create LWSFixationEvents
        from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
        from LWS.pre_processing_scripts.visual_angle_to_targets import \
            calculate_visual_angle_between_fixation_and_targets as calc_fixation_distance
        triggers = trial.get_behavioral_data().get(cnst.TRIGGER).values
        fixations_list = []
        for idxs in separate_event_idxs:
            fix = LWSFixationEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs], triggers=triggers[idxs])
            fix.visual_angle_to_target = calc_fixation_distance(fix=fix, trial=trial, sm=screen_monitor)
            fixations_list.append(fix)
        return fixations_list

    raise ValueError(f"Attempting to extract unknown event type {event_type}.")


def __extract_raw_event_arrays(trial: LWSTrial, event_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps, x, y = trial.get_raw_gaze_coordinates(eye='dominant')  # timestamps in milliseconds (floating-point, not integer)
    behavioral_data = trial.get_behavioral_data()
    is_event_colname = f"is_{event_type.lower()}"
    if is_event_colname not in behavioral_data.columns:
        raise ValueError(f"Behavioral Data does not contain column {is_event_colname}")
    is_event = behavioral_data.get(is_event_colname).values
    return timestamps, x, y, is_event
