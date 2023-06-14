# LWS PreProcessing Pipeline

import numpy as np
from typing import List, Tuple

import constants as cnst
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def gen_all_lws_events(trial: LWSTrial, screen_monitor: ScreenMonitor, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Generates all gaze events for the given trial and returns the events in a list sorted by their start time.
    """
    blink_events = gen_lws_gaze_events(trial, screen_monitor, cnst.BLINK)
    saccade_events = gen_lws_gaze_events(trial, screen_monitor, cnst.SACCADE)
    fixation_events = gen_lws_gaze_events(trial, screen_monitor, cnst.FIXATION)
    all_events = blink_events + saccade_events + fixation_events
    if drop_outliers:
        all_events = [event for event in all_events if not event.is_outlier]
    all_events.sort(key=lambda event: event.start_time)
    return all_events


def gen_lws_gaze_events(trial: LWSTrial, screen_monitor: ScreenMonitor,
                        event_type: str, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Identifies chunks of data that belong to the same event and creates a GazeEvent object for each chunk. Then, it
    sorts the events by their start time and returns the list of events.

    For more information see the generic implementation in GazeEvents.pre_processing_scripts.gen_gaze_events.py
    """
    event_type = event_type.lower()

    if event_type == cnst.FIXATION:
        # use LWS specific fixation events
        events_list = _gen_lws_fixation_events(trial=trial, screen_monitor=screen_monitor)
    else:
        # use generic gaze events
        from GazeEvents.scripts.gen_gaze_events import gen_gaze_events

        timestamps, x, y, is_event, _ = __extract_raw_event_arrays(trial=trial, event_type=event_type)
        events_list = gen_gaze_events(timestamps=timestamps, is_event=is_event, sampling_rate=trial.sampling_rate,
                                      event_type=event_type, x=x, y=y)

    if drop_outliers:
        events_list = [event for event in events_list if not event.is_outlier]
    events_list.sort(key=lambda event: event.start_time)
    return events_list


def _gen_lws_fixation_events(trial: LWSTrial, screen_monitor: ScreenMonitor) -> List[LWSFixationEvent]:
    """
    Specific implementation for generating LWSFixation events, that are a subclass of the generic FixationEvent that
    contains extra data fields that are unique to the LWS experiment.

    For the generic implementation see GazeEvents.pre_processing_scripts.gen_gaze_events.py
    """
    from Utils.array_utils import get_different_event_indices
    from LWS.pre_processing_scripts.distance_to_targets import calculate_visual_angle_between_fixation_and_targets

    timestamps, x, y, is_fixation, triggers = __extract_raw_event_arrays(trial=trial, event_type=cnst.FIXATION)
    separate_event_idxs = get_different_event_indices(is_fixation,
                                                      min_length=LWSFixationEvent.MINIMUM_TIMESTAMPS_FOR_EVENT)

    events_list = []
    for idxs in separate_event_idxs:
        fix = LWSFixationEvent(timestamps=timestamps[idxs],
                               sampling_rate=trial.sampling_rate,
                               x=x[idxs], y=y[idxs], triggers=triggers[idxs])
        fix.distance_to_target = calculate_visual_angle_between_fixation_and_targets(fix, trial, screen_monitor)
        events_list.append(fix)
    events_list.sort(key=lambda event: event.start_time)
    return events_list


def __extract_raw_event_arrays(trial: LWSTrial, event_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps, x, y = trial.get_raw_gaze_coordinates(eye='dominant')  # timestamps in milliseconds (floating-point, not integer)
    behavioral_data = trial.get_behavioral_data()
    is_event_colname = f"is_{event_type}"
    if is_event_colname not in behavioral_data.columns:
        raise ValueError(f"Behavioral Data does not contain column {is_event_colname}")
    is_event = behavioral_data.get(is_event_colname).values
    triggers = behavioral_data.get(cnst.TRIGGER).values
    return timestamps, x, y, is_event, triggers
