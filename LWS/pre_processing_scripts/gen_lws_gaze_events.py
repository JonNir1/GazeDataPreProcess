# LWS PreProcessing Pipeline

import numpy as np
from typing import List, Tuple

import constants as cnst
from Config import experiment_config as cnfg
import Utils.array_utils as au
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


def gen_all_lws_events(trial: LWSTrial, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Generates all gaze events for the given trial and returns the events in a list sorted by their start time.
    """
    blink_events = _gen_lws_gaze_events(GazeEventTypeEnum.BLINK, trial)
    saccade_events = _gen_lws_gaze_events(GazeEventTypeEnum.SACCADE, trial)
    fixation_events = _gen_lws_gaze_events(GazeEventTypeEnum.FIXATION, trial)
    all_events = blink_events + saccade_events + fixation_events
    if drop_outliers:
        all_events = [event for event in all_events if not event.is_outlier]
    all_events.sort(key=lambda event: event.start_time)
    return all_events


def _gen_lws_gaze_events(event_type: GazeEventTypeEnum, trial: LWSTrial) -> List[BaseGazeEvent]:
    """
    Identifies all chunks of data that belong to this type of event within the trial and creates a GazeEvent object
    for each chunk and returns the list of events.
    For more information see the generic implementation in GazeEvents.pre_processing_scripts.create_gaze_events.py

    :param event_type: type of event to extract
    :param trial: LWSTrial object

    :return: list of GazeEvent objects

    :raises: ValueError: if `event_type` is not one of 'blink', 'saccade' or 'fixation'
    """
    timestamps, x, y, p, is_event = __extract_raw_event_arrays(trial=trial, event_type=event_type)
    separate_event_idxs = au.get_chunk_indices(is_event, min_length=cnfg.DEFAULT_MINIMUM_SAMPLES_PER_EVENT)
    viewer_distance = trial.subject.distance_to_screen

    if event_type == GazeEventTypeEnum.BLINK:
        # create BlinkEvents
        from GazeEvents.BlinkEvent import BlinkEvent
        blinks_list = [BlinkEvent(timestamps=timestamps[idxs]) for idxs in au.get_chunk_indices(is_event)]
        return blinks_list

    if event_type == GazeEventTypeEnum.SACCADE:
        # create SaccadeEvents
        from GazeEvents.SaccadeEvent import SaccadeEvent
        saccades_list = [
            SaccadeEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs], viewer_distance=viewer_distance)
            for idxs in separate_event_idxs
        ]
        return saccades_list

    if event_type == GazeEventTypeEnum.FIXATION:
        # create LWSFixationEvents
        from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
        from LWS.pre_processing_scripts.visual_angle_to_targets import visual_angle_fixation_to_targets
        fixations_list = []
        for idxs in separate_event_idxs:
            fix = LWSFixationEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs], pupil=p[idxs],
                                   viewer_distance=viewer_distance, trial=trial)
            fix.visual_angle_to_targets = visual_angle_fixation_to_targets(fix=fix, trial=trial)
            fixations_list.append(fix)
        return fixations_list

    raise ValueError(f"Attempting to extract unknown event type {event_type}.")


def __extract_raw_event_arrays(
        trial: LWSTrial,
        event_type: GazeEventTypeEnum) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps, x, y, p = trial.get_raw_gaze_data(
        eye='dominant')  # timestamps in milliseconds (floating-point, not integer)
    behavioral_data = trial.get_behavioral_data()
    is_event_colname = f"is_{event_type.name.lower()}"
    if is_event_colname not in behavioral_data.columns:
        raise ValueError(f"Behavioral Data does not contain column {is_event_colname}")
    is_event = behavioral_data.get(is_event_colname)
    return timestamps, x, y, p, is_event
