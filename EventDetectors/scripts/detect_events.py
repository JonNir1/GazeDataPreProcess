import warnings as w
import numpy as np
from typing import Optional

from EventDetectors.BaseDetector import BaseDetector
from EventDetectors.scripts.gen_event_detector import gen_event_detector


def detect_event(x: np.ndarray, y: np.ndarray, sampling_rate: float,
                 detector_type: Optional[str] = None, detect_by: Optional[str] = None,
                 **detector_kwargs) -> np.ndarray:
    """
    Detects events using the specified detector type on the given gaze data.

    :param x: 1D or 2D array of x-coordinates of gaze data
    :param y: 1D or 2D array of y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events
    :param detector_type: name of the detector type to use. See gen_event_detector for details.
    :param detector_kwargs: keyword arguments for the specified detector type. see gen_event_detector for details.

    :return: is_event: array of booleans, where True indicates an event
    """
    detector = gen_event_detector(detector_type, sampling_rate, **detector_kwargs)
    is_event = _detect_event_generic(detector, x, y, detect_by)
    return is_event


def backfill_unidentified_samples(is_blink: np.ndarray, is_saccade: np.ndarray, is_fixation: np.ndarray,
                                  fill_with: Optional[str] = None) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Fills samples that were not identified as blinks, saccades or fixations with the specified value.
    :param is_blink, is_saccade, is_fixation: arrays of booleans, where True indicates an event
    :param fill_with: str; either "saccade", "fixation" or None. Controls how to fill unidentified samples.
            - If None: returns the events as identified by the respective detectors
            - If "saccade":
                -- if a saccade detector was specified, returns the events as identified by the saccade detector and
                    warns for unexpected usage
                --  if no saccade detector was specified, returns True for samples that were not identified as blinks or
                    fixations.
            - If "fixation":
                -- same behavior as "saccade", but for fixations.

    :return: new_is_blink, new_is_saccade, new_is_fixation: new arrays of booleans, where True indicates an event
    """
    if not fill_with:
        return is_blink, is_saccade, is_fixation
    if type(fill_with) != str:
        raise TypeError("stuff_with must be a string or None")
    fill_with = fill_with.lower()
    if fill_with not in ["saccade", "fixation", "fixations", "saccades"]:
        raise ValueError("stuff_with must be either 'saccade' or 'fixation'")

    new_is_blink = is_blink.copy()
    new_is_saccade = is_saccade.copy()
    new_is_fixation = is_fixation.copy()
    if fill_with == "saccade":
        new_is_saccade = np.logical_not(np.logical_or(new_is_blink, new_is_fixation))
    if fill_with == "fixation":
        new_is_fixation = np.logical_not(np.logical_or(new_is_blink, is_saccade))
    return new_is_blink, new_is_saccade, new_is_fixation


def detect_all_events(x: np.ndarray, y: np.ndarray,
                      sampling_rate: float,
                      detect_by: Optional[str] = None,
                      fill_with: Optional[str] = None,
                      **kwargs) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Detects blinks, saccades and fixations in the given gaze data (in that order).

    :param x: x-coordinates of gaze data
    :param y: y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events
    :param fill_with: str; either "saccade", "fixation" or None. Controls how to fill unidentified samples.
            - If None: returns the events as identified by the respective detectors
            - If "saccade":
                -- if a saccade detector was specified, returns the events as identified by the saccade detector and
                    warns for unexpected usage
                --  if no saccade detector was specified, returns True for samples that were not identified as blinks or
                    fixations.
            - If "fixation":
                -- same behavior as "saccade", but for fixations.

    :keyword arguments:
        - blink_detector_type: str; type of blink detector to use, None for no blink detection
        - saccade_detector_type: str; type of saccade detector to use, None for no saccade detection
        - fixation_detector_type: str; type of fixation detector to use, None for no fixation detection
        - See additional keyword arguments in the respective detection functions.

    :return: is_blink, is_saccade, is_fixation: arrays of booleans, where True indicates an event
    """
    blink_detector_type = kwargs.pop("blink_detector_type", None)
    is_blink = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                            detector_type=blink_detector_type,
                            detect_by=detect_by,
                            **kwargs)
    saccade_detector_type = kwargs.pop("saccade_detector_type", None)
    is_saccade = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                              detector_type=saccade_detector_type,
                              detect_by=detect_by,
                              **kwargs)
    fixation_detector_type = kwargs.pop("fixation_detector_type", None)
    is_fixation = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                               detector_type=fixation_detector_type,
                               detect_by=detect_by,
                               **kwargs)

    is_blink, is_saccade, is_fixation = backfill_unidentified_samples(is_blink, is_saccade, is_fixation, fill_with)
    return is_blink, is_saccade, is_fixation


def _detect_event_generic(detector: Optional[BaseDetector],
                          x: np.ndarray, y: np.ndarray,
                          detect_by: Optional[str] = None) -> np.ndarray:
    """
    Use the provided event detector to detect events in the given gaze data, either monocular or binocular.
    - If x and y are 1D arrays, the data is assumed to be monocular.
    - If x and y are 2D arrays, the data is assumed to be binocular and x[0], y[0] are assumed to be the left eye and
        x[1], y[1] are assumed to be the right eye. In this case, `detect_by` must be specified.

    :param detector: event detector to use
    :param x: 1D or 2D array of x-coordinates of gaze data
    :param y: 1D or 2D array of y-coordinates of gaze data
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events

    :return: array of booleans, where True indicates an event
    """
    if x.shape != y.shape:
        raise ValueError(f"x (shape: {x.shape}) and y (shape: {y.shape}) must have the same shape")

    if x.ndim == 1:
        # x.shape = (n,)
        if detector is None:
            return np.zeros_like(x, dtype=bool)
        return detector.detect_monocular(x, y)

    if x.ndim == 2 and x.shape[0] == 1:
        # x.shape = (1, n)
        if detector is None:
            return np.zeros_like(x[0], dtype=bool)
        return detector.detect_monocular(x[0], y[0])

    if x.ndim == 2 and x.shape[0] == 2:
        # x.shape = (2, n)
        if detector is None:
            return np.zeros_like(x[0], dtype=bool)
        if detect_by is None:
            raise ValueError("Binocular data provided, but detect_by not specified.")
        return detector.detect_binocular(x_l=x[0], y_l=y[0], x_r=x[1], y_r=y[1], detect_by=detect_by)

    raise ValueError(f"Invalid shape of x and y: {x.shape}")
