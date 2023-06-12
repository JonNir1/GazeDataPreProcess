import numpy as np
import pandas as pd
from typing import Optional, List

import constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent


def gen_gaze_events(event_type: str,
                    timestamps: np.ndarray, is_event: np.ndarray, sampling_rate: float,
                    x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> List[BaseGazeEvent]:
    """
    Splits `timestamps` to chunks of timestamps that are part of the same event, based on `is_event`. Then, for each
    chunk, creates a GazeEvent object of the given type and returns a list of all the events (in the same order as
    they appear in the data).

    :param event_type: type of event to extract. Must be one of 'blink', 'saccade' or 'fixation'
    :param timestamps: array of timestamps
    :param is_event: array of booleans indicating whether a sample is part of the event or not
    :param sampling_rate: float indicating the sampling rate of the data
    :param x: array of x coordinates, used when extracting saccades or fixations
    :param y: array of y coordinates, used when extracting saccades or fixations

    :return: list of GazeEvent objects

    :raises:
        - ValueError: if `timestamps` and `is_event` have different lengths
        - ValueError: if `event_type` is not one of 'blink', 'saccade' or 'fixation'
        - ValueError: if `x` and `y` are not provided when extracting saccades or fixations
    """
    if len(timestamps) != len(is_event):
        raise ValueError("Arrays of timestamps and is_event must have the same length")

    event_type = event_type.lower()
    allowed_event_types = [cnst.BLINK, cnst.SACCADE, cnst.FIXATION]
    if event_type not in allowed_event_types:
        raise ValueError(f"Attempting to extract unknown event type {event_type}. "
                         f"Argument event_type must be one of {str(allowed_event_types)}")

    if event_type in [cnst.SACCADE, cnst.FIXATION] and (x is None or y is None):
        raise ValueError(f"Attempting to extract {event_type} without providing x and y coordinates")

    different_event_idxs = _split_samples_between_events(is_event)
    if event_type == cnst.BLINK:
        from GazeEvents.BlinkEvent import BlinkEvent
        events_list = [BlinkEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate)
                       for idxs in different_event_idxs]
        return events_list

    if event_type == cnst.SACCADE:
        from GazeEvents.SaccadeEvent import SaccadeEvent
        events_list = [SaccadeEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate,
                                    x=x[idxs], y=y[idxs]) for idxs in different_event_idxs]
        return events_list

    if event_type == cnst.FIXATION:
        from GazeEvents.FixationEvent import FixationEvent
        events_list = [FixationEvent(timestamps=timestamps[idxs], sampling_rate=sampling_rate,
                                     x=x[idxs], y=y[idxs]) for idxs in different_event_idxs]
        return events_list

    # should never get here
    raise ValueError(f"Attempting to extract unknown event type {event_type}. "
                     f"Argument event_type must be one of {str(allowed_event_types)}")


def gen_gaze_events_summary(event_type: str,
                            timestamps: np.ndarray, is_event: np.ndarray, sampling_rate: float,
                            x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Splits `timestamps` to chunks of timestamps that are part of the same event, based on `is_event`. Then, for each
    chunk, creates a GazeEvent object of the given type. Finally, returns a pandas DataFrame with the events' summary
    information (one row per event).
    See further documentation in `gen_gaze_events`.

    :return: pandas DataFrame with the events' summary information
    """
    events_list = gen_gaze_events(event_type=event_type, timestamps=timestamps, is_event=is_event,
                                  sampling_rate=sampling_rate, x=x, y=y)
    return pd.concat([event.to_series() for event in events_list], axis=1).T


def _split_samples_between_events(is_event: np.ndarray) -> List[np.ndarray]:
    # returns a list of arrays, each array contains the indices of the samples that belong to the same event
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    return different_event_idxs
