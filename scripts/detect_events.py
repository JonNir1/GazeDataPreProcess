import numpy as np

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector
from EventDetectors.EngbertSaccadeDetector import DEFAULT_DERIVATION_WINDOW_SIZE, DEFAULT_LAMBDA_NOISE_THRESHOLD
from EventDetectors.IVTFixationDetector import DEFAULT_VELOCITY_THRESHOLD


def detect_all_events(x: np.ndarray, y: np.ndarray,
                      sampling_rate: float, inter_event_time: float,
                      **kwargs) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Detects blinks, saccades and fixations in the given gaze data (in that order).

    :param x: x-coordinates of gaze data
    :param y: y-coordinates of gaze data
    :param sampling_rate: sampling rate of the data in Hz
    :param inter_event_time: minimal time between two events in ms

    :keyword blink_detector_type: type of blink detector to use, None for no blink detection
    :keyword blink_min_duration: minimal duration of a blink in ms

    :return: is_blink, is_saccade, is_fixation: arrays of booleans, where True indicates an event
    """
    # detect blinks:
    blink_detector_type = kwargs.get("blink_detector_type", None)
    if blink_detector_type:
        min_duration = kwargs.get("blink_min_duration", conf.BLINK_MINIMUM_DURATION)
        blink_kwargs = {
            "missing_value": kwargs.get("missing_value", conf.MISSING_VALUE)
        }
        blink_detector = _get_event_detector(blink_detector_type, min_duration=min_duration,
                                             sampling_rate=sampling_rate, inter_event_time=inter_event_time,
                                             **blink_kwargs)
        is_blink = blink_detector.detect(x, y)
    else:
        is_blink = np.zeros_like(x, dtype=bool)

    # detect saccades:
    saccade_detector_type = kwargs.get("saccade_detector_type", None)
    if saccade_detector_type:
        min_duration = kwargs.get("saccade_min_duration", conf.SACCADE_MINIMUM_DURATION)
        saccade_kwargs = {
            "derivation_window_size": kwargs.get("derivation_window_size", DEFAULT_DERIVATION_WINDOW_SIZE),
            "lambda_noise_threshold": kwargs.get("lambda_noise_threshold", DEFAULT_LAMBDA_NOISE_THRESHOLD)
        }
        saccade_detector = _get_event_detector(saccade_detector_type, min_duration=min_duration,
                                               sampling_rate=sampling_rate, inter_event_time=inter_event_time,
                                               **saccade_kwargs)
        is_saccade = saccade_detector.detect(x, y)
    else:
        is_saccade = np.zeros_like(x, dtype=bool)

    # detect fixations:
    fixation_detector_type = kwargs.get("fixation_detector_type", None)
    if fixation_detector_type:
        min_duration = kwargs.get("fixation_min_duration", conf.FIXATION_MINIMUM_DURATION)
        fixation_kwargs = {
            "velocity_threshold": kwargs.get("velocity_threshold", DEFAULT_VELOCITY_THRESHOLD)
        }
        fixation_detector = _get_event_detector(fixation_detector_type, min_duration=min_duration,
                                                sampling_rate=sampling_rate, inter_event_time=inter_event_time,
                                                **fixation_kwargs)
        is_fixation = fixation_detector.detect(x, y)
    else:
        is_fixation = np.zeros_like(x, dtype=bool)
    return is_blink, is_saccade, is_fixation


def _get_event_detector(detector_type: str, min_duration: float, sampling_rate: float,
                        inter_event_time: float, **kwargs) -> BaseDetector:
    """
    Creates an event detector of the given type.
    :param detector_type: type of the event detector
    :param min_duration: minimal duration of an event in ms
    :param sampling_rate: sampling rate of the data in Hz
    :param inter_event_time: minimal time between two events in ms

    :keyword:
        - missing_value: default value indicating missing data              used by MissingDataBlinkDetector
        - derivation_window_size: window size for derivation                used by EngbertSaccadeDetector
        - lambda_noise_threshold: threshold for noise                       used by EngbertSaccadeDetector
        - velocity_threshold: threshold for velocity                        used by IVTFixationDetector

    :return: is_event: array of booleans, where True indicates an event

    :raises ValueError: if a required keyword argument is not specified
    """
    if detector_type.lower() == "missing data" or detector_type.lower() == "missing_data":
        from EventDetectors.MissingDataBlinkDetector import MissingDataBlinkDetector
        missing_value = kwargs.get("missing_value", None)
        if missing_value is None:
            raise ValueError("Missing value for MissingDataBlinkDetector not specified.")
        return MissingDataBlinkDetector(sr=sampling_rate, iet=inter_event_time, min_duration=min_duration,
                                        missing_value=missing_value)

    if detector_type.lower() == "pupil size" or detector_type.lower() == "pupil_size":
        from EventDetectors.PupilSizeBlinkDetector import PupilSizeBlinkDetector
        return PupilSizeBlinkDetector()

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

