# LWS PreProcessing Pipeline

import numpy as np
from typing import List

import constants as cnst
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from LWS.scripts.distance_to_targets import calculate_angular_target_distance_for_fixation


def extract_all_events(trial: LWSTrial, screen_monitor: ScreenMonitor, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Extracts all events from the given data and returns a list of GazeEvent objects.
    """
    blink_events = extract_event(trial, screen_monitor, cnst.BLINK)
    saccade_events = extract_event(trial, screen_monitor, cnst.SACCADE)
    fixation_events = extract_event(trial, screen_monitor, cnst.FIXATION)
    all_events = blink_events + saccade_events + fixation_events
    if drop_outliers:
        all_events = [event for event in all_events if not event.is_outlier]
    all_events.sort(key=lambda event: event.start_time)
    return all_events


def extract_event(trial: LWSTrial, screen_monitor: ScreenMonitor,
                  event_type: str, drop_outliers: bool = False) -> List[BaseGazeEvent]:
    """
    Extracts events of the given type from the given data and returns a list of BaseGazeEvent objects.
    """
    event_type = event_type.lower()
    is_event_colname = f"is_{event_type}"
    allowed_event_types = [cnst.BLINK, cnst.SACCADE, cnst.FIXATION]
    behavioral_data = trial.get_behavioral_data()
    if event_type not in allowed_event_types:
        raise NotImplementedError(f"event_type must be one of {str(allowed_event_types)}, got {event_type}")
    if is_event_colname not in behavioral_data.columns:
        raise ValueError(f"Behavioral Data does not contain column {is_event_colname}")

    timestamps, x, y = trial.get_raw_gaze_coordinates()  # timestamps in milliseconds (floating-point, not integer)
    is_event = behavioral_data.get(is_event_colname).values
    if len(timestamps) != len(is_event):
        raise ValueError(f"Arrays of \'timestamps\' and \'{is_event_colname}\' must have the same length")
    separate_event_idxs = _split_samples_between_events(is_event)
    events_list = []

    if event_type == cnst.BLINK:
        from GazeEvents.BlinkEvent import BlinkEvent
        events_list = [BlinkEvent(timestamps=timestamps[idxs], sampling_rate=trial.sampling_rate)
                       for idxs in separate_event_idxs]

    if event_type == cnst.SACCADE:
        from GazeEvents.SaccadeEvent import SaccadeEvent
        events_list = [SaccadeEvent(timestamps=timestamps[idxs], sampling_rate=trial.sampling_rate,
                                    x=x[idxs], y=y[idxs]) for idxs in separate_event_idxs]

    if event_type == cnst.FIXATION:
        from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
        events_list = []
        for idxs in separate_event_idxs:
            fix = LWSFixationEvent(timestamps=timestamps[idxs],
                                   sampling_rate=trial.sampling_rate,
                                   x=x[idxs], y=y[idxs])
            fix.distance_to_target = calculate_angular_target_distance_for_fixation(fix, trial, screen_monitor)
            events_list.append(fix)

    if drop_outliers:
        events_list = [event for event in events_list if not event.is_outlier]
    events_list.sort(key=lambda event: event.start_time)
    return events_list


def _split_samples_between_events(is_event: np.ndarray) -> List[np.ndarray]:
    # returns a list of arrays, each array contains the indices of the samples that belong to the same event
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    return different_event_idxs
