import warnings as w
import numpy as np
from typing import Optional

import experiment_config as cnfg
from EventDetectors.BaseDetector import BaseDetector


def detect_all_events(x: np.ndarray, y: np.ndarray,
                      sampling_rate: float,
                      detect_by: Optional[str] = None,
                      stuff_with: Optional[str] = None,
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
    :param stuff_with: str; either "saccade", "fixation" or None. Controls how to fill unidentified samples.
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
    is_blink = detect_blinks(blink_detector_type, x, y, sampling_rate, detect_by, **kwargs)

    saccade_detector_type = kwargs.pop("saccade_detector_type", None)
    is_saccade = detect_saccades(saccade_detector_type, x, y, sampling_rate, detect_by, **kwargs)

    fixation_detector_type = kwargs.pop("fixation_detector_type", None)
    is_fixation = detect_fixations(fixation_detector_type, x, y, sampling_rate, detect_by, **kwargs)

    # classify unidentified samples with value specified in stuff_with:
    if not stuff_with:
        return is_blink, is_saccade, is_fixation
    if type(stuff_with) != str:
        raise TypeError("stuff_with must be a string or None")
    if stuff_with.lower() not in ["saccade", "fixation"]:
        raise ValueError("stuff_with must be either 'saccade' or 'fixation'")

    if stuff_with.lower() == "saccade":
        if saccade_detector_type:
            w.warn("WARNING: ignoring stuff_with='saccade' when a saccade detector is specified")
            return is_blink, is_saccade, is_fixation
        is_saccade = np.logical_not(np.logical_or(is_blink, is_fixation))

    if stuff_with.lower() == "fixation":
        if fixation_detector_type:
            w.warn("WARNING: ignoring stuff_with='fixation' when a fixation detector is specified")
            return is_blink, is_saccade, is_fixation
        is_fixation = np.logical_not(np.logical_or(is_blink, is_saccade))
    return is_blink, is_saccade, is_fixation


def detect_blinks(blink_detector_type: Optional[str],
                  x: np.ndarray, y: np.ndarray,
                  sampling_rate: float, detect_by: Optional[str] = None,
                  **kwargs) -> np.ndarray:
    """
    Detects blinks in the given gaze data, based on the specified blink detector type.

    :param blink_detector_type: type of blink detector to use, None for no blink detection
    :param x: 1D or 2D array of x-coordinates of gaze data
    :param y: 1D or 2D array of y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms; default: 5 ms
        - blink_min_duration: minimal duration of a blink in ms; default: 50 ms
        - missing_value: default value indicating missing data, used by MissingDataBlinkDetector; default: np.nan

    :return: array of booleans, where True indicates a blink
    """
    if not blink_detector_type:
        return np.zeros_like(x, dtype=bool)
    iet = kwargs.get("inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME)
    min_duration = kwargs.get("blink_min_duration", cnfg.DEFAULT_BLINK_MINIMUM_DURATION)
    blink_kwargs = {
        "missing_value": kwargs.get("missing_value", cnfg.DEFAULT_MISSING_VALUE)
    }
    blink_detector = _get_event_detector(blink_detector_type,
                                         min_duration=min_duration,
                                         sampling_rate=sampling_rate,
                                         inter_event_time=iet,
                                         **blink_kwargs)
    is_blink = __detect_event_generic(detector=blink_detector, x=x, y=y, detect_by=detect_by)
    return is_blink


def detect_saccades(saccade_detector_type: Optional[str],
                    x: np.ndarray, y: np.ndarray,
                    sampling_rate: float, detect_by: Optional[str] = None,
                    **kwargs) -> np.ndarray:
    """
    Detects saccades in the given gaze data, based on the specified saccades detector type.

    :param saccade_detector_type: type of saccade detector to use, None for no saccade detection
    :param x: 1D or 2D array of x-coordinates of gaze data
    :param y: 1D or 2D array of y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms; default: 5 ms
        - saccade_min_duration: minimal duration of a blink in ms;  default: 5 ms
        - derivation_window_size: window size for derivation in ms; default: 3 ms
        - lambda_noise_threshold: threshold for lambda noise;       default: 5

    :return: array of booleans, where True indicates a saccade
    """
    if not saccade_detector_type:
        return np.zeros_like(x, dtype=bool)
    from EventDetectors.EngbertSaccadeDetector import DEFAULT_DERIVATION_WINDOW_SIZE, DEFAULT_LAMBDA_NOISE_THRESHOLD
    iet = kwargs.get("inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME)
    min_duration = kwargs.get("saccade_min_duration", cnfg.DEFAULT_SACCADE_MINIMUM_DURATION)
    saccade_kwargs = {
        "derivation_window_size": kwargs.get("derivation_window_size", DEFAULT_DERIVATION_WINDOW_SIZE),
        "lambda_noise_threshold": kwargs.get("lambda_noise_threshold", DEFAULT_LAMBDA_NOISE_THRESHOLD)
    }
    saccade_detector = _get_event_detector(saccade_detector_type, min_duration=min_duration,
                                           sampling_rate=sampling_rate, inter_event_time=iet,
                                           **saccade_kwargs)
    is_saccade = __detect_event_generic(detector=saccade_detector, x=x, y=y, detect_by=detect_by)
    return is_saccade


def detect_fixations(fixation_detector_type: Optional[str],
                     x: np.ndarray, y: np.ndarray,
                     sampling_rate: float,
                     detect_by: Optional[str] = None,
                     **kwargs) -> np.ndarray:
    """
    Detects fixations in the given gaze data, based on the specified fixation detector type.

    :param fixation_detector_type: type of fixation detector to use, None for no fixation detection
    :param x: 1D or 2D array of x-coordinates of gaze data
    :param y: 1D or 2D array of y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms; default: 5 ms
        - fixation_min_duration: minimal duration of a blink in ms;         default: 55 ms
        - velocity_threshold: maximal velocity allowed within a fixation;   default: 30 deg/s

    :return: array of booleans, where True indicates a saccade
    """
    if not fixation_detector_type:
        return np.zeros_like(x, dtype=bool)
    iet = kwargs.get("inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME)
    min_duration = kwargs.get("fixation_min_duration", cnfg.DEFAULT_FIXATION_MINIMUM_DURATION)
    fixation_kwargs = {
        "velocity_threshold": kwargs.get("velocity_threshold", cnfg.DEFAULT_FIXATION_MAX_VELOCITY)
    }
    fixation_detector = _get_event_detector(fixation_detector_type, min_duration=min_duration,
                                            sampling_rate=sampling_rate, inter_event_time=iet,
                                            **fixation_kwargs)
    is_fixation = __detect_event_generic(detector=fixation_detector, x=x, y=y, detect_by=detect_by)
    return is_fixation


def _get_event_detector(detector_type: str, min_duration: float, sampling_rate: float,
                        inter_event_time: float, **kwargs) -> BaseDetector:
    """
    Creates an event detector of the given type.
    :param detector_type: type of the event detector
    :param min_duration: minimal duration of an event in ms
    :param sampling_rate: sampling rate of the data in Hz
    :param inter_event_time: minimal time between two events in ms

    :keyword arguments:
        - missing_value: default value indicating missing data              used by MissingDataBlinkDetector
        - derivation_window_size: window size for derivation                used by EngbertSaccadeDetector
        - lambda_noise_threshold: threshold for noise                       used by EngbertSaccadeDetector
        - velocity_threshold: threshold for velocity                        used by IVTFixationDetector
    :return: is_event: array of booleans, where True indicates an event
    :raises ValueError: if a required keyword argument is not specified
    """
    if detector_type.lower() == "missing data" or detector_type.lower() == "missing_data":
        from EventDetectors.MissingDataBlinkDetector import MissingDataBlinkDetector
        missing_value = kwargs.get("missing_value", cnfg.DEFAULT_MISSING_VALUE)
        return MissingDataBlinkDetector(sr=sampling_rate, iet=inter_event_time, min_duration=min_duration,
                                        missing_value=missing_value)

    if detector_type.lower() == "pupil size" or detector_type.lower() == "pupil_size":
        from EventDetectors.PupilSizeBlinkDetector import PupilSizeBlinkDetector
        return PupilSizeBlinkDetector(sr=sampling_rate, iet=inter_event_time, min_duration=min_duration)

    if detector_type.lower() == "engbert":
        from EventDetectors.EngbertSaccadeDetector import EngbertSaccadeDetector
        derivation_window_size = kwargs.get("derivation_window_size", None)
        if derivation_window_size is None:
            raise ValueError("Derivation window size for EngbertSaccadeDetector not specified.")
        lambda_noise_threshold = kwargs.get("lambda_noise_threshold", None)
        if lambda_noise_threshold is None:
            raise ValueError("Lambda noise threshold for EngbertSaccadeDetector not specified.")
        return EngbertSaccadeDetector(sr=sampling_rate, iet=inter_event_time, min_duration=min_duration,
                                      derivation_window_size=derivation_window_size,
                                      lambda_noise_threshold=lambda_noise_threshold)

    if detector_type.lower() == "ivt":
        from EventDetectors.IVTFixationDetector import IVTFixationDetector
        velocity_threshold = kwargs.get("velocity_threshold", None)
        if velocity_threshold is None:
            raise ValueError("Velocity threshold for IVTFixationDetector not specified.")
        return IVTFixationDetector(sr=sampling_rate, min_duration=min_duration,
                                   iet=inter_event_time, velocity_threshold=velocity_threshold)

    if detector_type.lower() == "idt":
        from EventDetectors.IDTFixationDetector import IDTFixationDetector
        return IDTFixationDetector(sr=sampling_rate, min_duration=min_duration,
                                   iet=inter_event_time)

    # reached here if the detector type is unknown
    raise ValueError("Unknown event detector type: {}".format(detector_type))


def __detect_event_generic(detector: BaseDetector,
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
        return detector.detect_monocular(x, y)
    if x.ndim == 2 and x.shape[0] == 1:
        # x.shape = (1, n)
        return detector.detect_monocular(x[0], y[0])
    if x.ndim == 2 and x.shape[0] == 2:
        # x.shape = (2, n)
        if detect_by is None:
            raise ValueError("Binocular data provided, but detect_by not specified.")
        return detector.detect_binocular(x_l=x[0], y_l=y[0], x_r=x[1], y_r=y[1], detect_by=detect_by)
    raise ValueError(f"Invalid shape of x and y: {x.shape}")
