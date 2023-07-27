import numpy as np
import pandas as pd
from typing import Optional, List

import constants as cnst
from Config import experiment_config as cnfg
import Utils.array_utils as au
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


# TODO: use GazeEventTypeEnum instead of strings
def create_gaze_events(event_type: str, timestamps: np.ndarray, is_event: np.ndarray,
                       x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                       p: Optional[np.ndarray] = None, viewer_distance: Optional[float] = None) -> List[BaseGazeEvent]:
    """
    Splits `timestamps` to chunks of timestamps that are part of the same event, based on `is_event`. Then, for each
    chunk, creates a GazeEvent object of the given type and returns a list of all the events (ordered by start time).

    :param event_type: type of event to extract. Must be one of 'blink', 'saccade' or 'fixation'
    :param timestamps: array of timestamps
    :param is_event: array of booleans indicating whether a sample is part of the event or not
    :param x: array of x coordinates, used when extracting saccades or fixations
    :param y: array of y coordinates, used when extracting saccades or fixations
    :param p: array of pupil sizes, used when extracting fixations
    :param viewer_distance: distance of the viewer from the screen, used when extracting saccades or fixations

    :return: list of GazeEvent objects

    :raises:
        - ValueError: if `timestamps` and `is_event` have different lengths
        - ValueError: if `event_type` is not one of 'blink', 'saccade' or 'fixation'
        - ValueError: if `x` and `y` are not provided when extracting saccades or fixations
    """
    if len(timestamps) != len(is_event):
        raise ValueError("Arrays of `timestamps` and `is_event` must have the same length")

    event_type = event_type.lower()
    allowed_event_types = [et.name.lower() for et in GazeEventTypeEnum]
    if event_type not in allowed_event_types:
        raise ValueError(f"Attempting to extract unknown event type {event_type}. "
                         f"Argument event_type must be one of {str(allowed_event_types)}")
    if event_type in [cnst.SACCADE, cnst.FIXATION] and (x is None or y is None):
        raise ValueError(f"Attempting to extract {event_type} without providing x and y coordinates")

    different_event_idxs = au.get_chunk_indices(is_event, min_length=cnfg.DEFAULT_MINIMUM_SAMPLES_PER_EVENT)
    events_list = []
    if event_type == cnst.BLINK:
        from GazeEvents.BlinkEvent import BlinkEvent
        events_list = [BlinkEvent(timestamps=timestamps[idxs])
                       for idxs in different_event_idxs]

    if event_type == cnst.SACCADE:
        from GazeEvents.SaccadeEvent import SaccadeEvent
        events_list = [SaccadeEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs], viewer_distance=viewer_distance)
                       for idxs in different_event_idxs]

    if event_type == cnst.FIXATION:
        from GazeEvents.FixationEvent import FixationEvent
        events_list = [FixationEvent(timestamps=timestamps[idxs], x=x[idxs], y=y[idxs],
                                     pupil=p[idxs], viewer_distance=viewer_distance)
                       for idxs in different_event_idxs]

    events_list.sort(key=lambda event: event.start_time)
    return events_list


def gen_gaze_events_summary(event_type: str,
                            timestamps: np.ndarray, is_event: np.ndarray,
                            x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Splits `timestamps` to chunks of timestamps that are part of the same event, based on `is_event`. Then, for each
    chunk, creates a GazeEvent object of the given type. Finally, returns a pandas DataFrame with the events' summary
    information (one row per event).
    See further documentation in `gen_gaze_events`.

    :return: pandas DataFrame with the events' summary information
    """
    events_list = create_gaze_events(event_type=event_type, timestamps=timestamps, is_event=is_event, x=x, y=y)
    return pd.concat([event.to_series() for event in events_list], axis=1).T
