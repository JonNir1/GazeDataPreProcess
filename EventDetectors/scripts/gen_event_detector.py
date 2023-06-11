from typing import Optional, Tuple, Dict, Any

import experiment_config as cnfg
from EventDetectors.BaseDetector import BaseDetector
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector
from EventDetectors.BaseSaccadeDetector import BaseSaccadeDetector
from EventDetectors.BaseFixationDetector import BaseFixationDetector


def gen_event_detector(detector_type: Optional[str], sampling_rate: float, **kwargs) -> Optional[BaseDetector]:
    """
    Creates an event detector of the given type.
    :param detector_type: type of the event detector
    :param sampling_rate: sampling rate of the data in Hz

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms           used by all detectors
        - min_duration: minimal duration of an event in ms                  used by all detectors
        - missing_value: default value indicating missing data              used by MissingDataBlinkDetector
        - derivation_window_size: window size for derivation                used by EngbertSaccadeDetector
        - lambda_noise_threshold: threshold for noise                       used by EngbertSaccadeDetector
        - velocity_threshold: threshold for velocity                        used by IVTFixationDetector

    :return: BaseDetector object or None if detector_type is None

    :raises ValueError: if a required keyword argument is not specified
    """
    if detector_type is None:
        return None
    detector_type = detector_type.lower()
    if detector_type in ["missing data", "missing_data", "pupil size", "pupil_size"]:
        return gen_blink_detector(detector_type, sampling_rate, **kwargs)
    if detector_type in ["engbert"]:
        return gen_saccade_detector(detector_type, sampling_rate, **kwargs)
    if detector_type in ["ivt", "idt"]:
        return gen_fixation_detector(detector_type, sampling_rate, **kwargs)
    # reached here if the detector type is unknown
    raise ValueError("Unknown event detector type: {}".format(detector_type))


def gen_blink_detector(blink_detector_type: str, sampling_rate: float, **kwargs) -> BaseBlinkDetector:
    """
    Creates a blink detector of the given type, or raises a ValueError if the given type is unknown.
    :param blink_detector_type: type of the blink detector
    :param sampling_rate: sampling rate of the data in Hz

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms
        - blink_min_duration: minimal duration of a blink in ms
        - missing_value: default value indicating missing data (used by MissingDataBlinkDetector)

    :return: a BaseBlinkDetector object
    """
    blink_detector_type = blink_detector_type.lower()
    iet, min_duration, blink_kwargs = __extract_arguments_by_event_type("blink", **kwargs)

    if blink_detector_type == "missing data" or blink_detector_type == "missing_data":
        from EventDetectors.MissingDataBlinkDetector import MissingDataBlinkDetector
        return MissingDataBlinkDetector(sr=sampling_rate, iet=iet, min_duration=min_duration,
                                        missing_value=blink_kwargs["missing_value"])

    if blink_detector_type.lower() == "pupil size" or blink_detector_type.lower() == "pupil_size":
        from EventDetectors.PupilSizeBlinkDetector import PupilSizeBlinkDetector
        return PupilSizeBlinkDetector(sr=sampling_rate, iet=iet, min_duration=min_duration)

    raise ValueError("Unknown blink detector type: {}".format(blink_detector_type))


def gen_saccade_detector(saccade_detector_type: str, sampling_rate: float, **kwargs) -> BaseSaccadeDetector:
    """
    Creates a saccade detector of the given type, or raises a ValueError if the given type is unknown.
    :param saccade_detector_type:
    :param sampling_rate:

    :keyword arguments:
        - inter_event_time: minimal time between two events in ms
        - saccade_min_duration: minimal duration of an event in ms
        - derivation_window_size: window size for derivation (used by EngbertSaccadeDetector)
        - lambda_noise_threshold: threshold for noise (used by EngbertSaccadeDetector)

    :return:
    """
    saccade_detector_type = saccade_detector_type.lower()
    iet, min_duration, saccade_kwargs = __extract_arguments_by_event_type("saccade", **kwargs)

    if saccade_detector_type == "engbert":
        from EventDetectors.EngbertSaccadeDetector import EngbertSaccadeDetector
        return EngbertSaccadeDetector(sr=sampling_rate, iet=iet, min_duration=min_duration,
                                      derivation_window_size=saccade_kwargs["derivation_window_size"],
                                      lambda_noise_threshold=saccade_kwargs["lambda_noise_threshold"])

    raise ValueError("Unknown saccade detector type: {}".format(saccade_detector_type))


def gen_fixation_detector(fixation_detector_type: str, sampling_rate: float, **kwargs) -> BaseFixationDetector:
    """
    Creates a fixation detector of the given type, or raises a ValueError if the given type is unknown.
    :param fixation_detector_type:
    :param sampling_rate:
    :param kwargs:
    :return:
    """
    fixation_detector_type = fixation_detector_type.lower()
    iet, min_duration, fixation_kwargs = __extract_arguments_by_event_type("fixation", **kwargs)

    if fixation_detector_type == "ivt":
        from EventDetectors.IVTFixationDetector import IVTFixationDetector
        return IVTFixationDetector(sr=sampling_rate, min_duration=min_duration, iet=iet,
                                   velocity_threshold=fixation_kwargs["velocity_threshold"])

    if fixation_detector_type == "idt":
        from EventDetectors.IDTFixationDetector import IDTFixationDetector
        return IDTFixationDetector(sr=sampling_rate, min_duration=min_duration, iet=iet)

    raise ValueError("Unknown fixation detector type: {}".format(fixation_detector_type))


def __extract_arguments_by_event_type(event_type: str, **kwargs) -> Tuple[float, float, Dict[str, Any]]:
    """
    Extracts the arguments for the event detector of the given event type from the given keyword arguments.
    :param event_type: either "blink", "saccade", or "fixation"
    :param kwargs: input keyword arguments for the event detector

    :return: tuple of (inter_event_time, min_duration, other_event_detector_kwargs)

    :raises ValueError: if the given event type is not supported
    """
    event_type = event_type.lower()
    iet = kwargs.get("inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME)

    if event_type == "blink":
        min_duration = kwargs.get("blink_min_duration", cnfg.DEFAULT_BLINK_MINIMUM_DURATION)
        blink_kwargs = {"missing_value": kwargs.get("missing_value", cnfg.DEFAULT_MISSING_VALUE)}
        return iet, min_duration, blink_kwargs

    if event_type == "saccade":
        from EventDetectors.EngbertSaccadeDetector import DEFAULT_DERIVATION_WINDOW_SIZE, DEFAULT_LAMBDA_NOISE_THRESHOLD
        min_duration = kwargs.get("saccade_min_duration", cnfg.DEFAULT_SACCADE_MINIMUM_DURATION)
        saccade_kwargs = {
            "derivation_window_size": kwargs.get("derivation_window_size", DEFAULT_DERIVATION_WINDOW_SIZE),
            "lambda_noise_threshold": kwargs.get("lambda_noise_threshold", DEFAULT_LAMBDA_NOISE_THRESHOLD)
        }
        return iet, min_duration, saccade_kwargs

    if event_type == "fixation":
        min_duration = kwargs.get("fixation_min_duration", cnfg.DEFAULT_FIXATION_MINIMUM_DURATION)
        fixation_kwargs = {
            "velocity_threshold": kwargs.get("velocity_threshold", cnfg.DEFAULT_FIXATION_MAX_VELOCITY)
        }
        return iet, min_duration, fixation_kwargs

    raise ValueError("Unknown event type: {}".format(event_type))
