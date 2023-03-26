import numpy as np
import pandas as pd
from typing import Optional, List

from GazeEvents.BaseEvent import BaseEvent


def extract_events_to_list(event_type: str,
                           timestamps: np.ndarray, is_event: np.ndarray, sampling_rate: float,
                           x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> List[BaseEvent]:
    """
    Extracts events of the given type from the given data and returns a list of Event objects.
    :param event_type: type of event to extract. Must be one of 'blink', 'saccade' or 'fixation'
    :param timestamps: array of timestamps
    :param is_event: array of booleans indicating whether a sample is part of the event or not
    :param sampling_rate: float indicating the sampling rate of the data
    :param x: array of x coordinates
    :param y: array of y coordinates

    :return: list of GazeEvent objects
    """
    if len(timestamps) != len(is_event):
        raise ValueError("Arrays of timestamps and is_event must have the same length")
    different_event_idxs = _split_samples_between_events(is_event)

    if event_type.lower() == "blink" or event_type.lower() == "blinks":
        from GazeEvents.BlinkEvent import BlinkEvent
        events_list = [BlinkEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate)
                       for idxs in different_event_idxs]
        return events_list

    if event_type.lower() == "saccade" or event_type.lower() == "saccades":
        if x is None or y is None:
            raise ValueError("x and y must be provided when extracting saccades")
        from GazeEvents.SaccadeEvent import SaccadeEvent
        events_list = [SaccadeEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate,
                                    x=x[idxs], y=y[idxs]) for idxs in different_event_idxs]
        return events_list

    if event_type.lower() == "fixation" or event_type.lower() == "fixations":
        if x is None or y is None:
            raise ValueError("x and y must be provided when extracting fixations")
        from GazeEvents.FixationEvent import FixationEvent
        events_list = [FixationEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate,
                                     x=x[idxs], y=y[idxs]) for idxs in different_event_idxs]
        return events_list

    raise ValueError(
        f"Unknown event type {event_type}. Argument event_type must be one of 'blink', 'saccade' or 'fixation'")


def extract_events_to_dataframe(event_type: str,
                                timestamps: np.ndarray, is_event: np.ndarray, sampling_rate: float,
                                x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Extracts events of the given type from the given data and returns a pandas DataFrame with the events' information.
    :param event_type: type of event to extract. Must be one of 'blink', 'saccade' or 'fixation'
    :param timestamps: array of timestamps
    :param is_event: array of booleans indicating whether a sample is part of the event or not
    :param sampling_rate: float indicating the sampling rate of the data
    :param x: array of x coordinates
    :param y: array of y coordinates

    :return: pandas DataFrame with the events' summary information
    """
    events_list = extract_events_to_list(event_type=event_type, timestamps=timestamps, is_event=is_event,
                                         sampling_rate=sampling_rate, x=x, y=y)
    return pd.concat([event.to_series() for event in events_list], axis=1).T


def _split_samples_between_events(is_event: np.ndarray) -> List[np.ndarray]:
    """
    returns a list of arrays, each array contains the indices of the samples that belong to the same event
    """
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    return different_event_idxs
