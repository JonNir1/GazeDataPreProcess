# LWS PreProcessing Pipeline

import numpy as np
from typing import Tuple

from Config import experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from EventDetectors.scripts.detect_events import detect_event, backfill_unidentified_samples
from EventDetectors.EngbertSaccadeDetector import DEFAULT_DERIVATION_WINDOW_SIZE, DEFAULT_LAMBDA_NOISE_THRESHOLD


def detect_all_events(trial: LWSTrial, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects all types of events in the given trial: fixations, saccades, and blinks.
    Returns a tuple of three boolean arrays, one for each type of event.
    :param trial: The trial to detect events in.

    keyword arguments:
    - eye: str; Controls which eye to use for detection. Either "left", "right", "dominant" or "both" (default "both")
    - fill_with: Controls how to fill unidentified samples. Either "saccade", "fixation" or None (default "fixation").

    :blink keyword arguments:
        - blink_detector_type: str; type of blink detector to use:
            -- default: "missing data" (MissingDataBlinkDetector)
            -- None for no blink detection
        - blink_detect_by: str; controls how to detect blinks if using binocular data (default: 'most').
            see BaseDetector.detect_binocular() for details.
        - blink_inter_event_time: int; minimal time between two events in ms
        - blink_min_duration: int; minimal duration of a blink in ms
        - missing_value: int; value that indicates missing data

    :saccade keyword arguments:
        - saccade_detector_type: str; type of saccade detector to use:
            -- default: "engbert" (EngbertSaccadeDetector)
            -- None for no saccade detection
        - saccade_detect_by: str; controls how to detect saccades if using binocular data (default: 'both').
            see BaseDetector.detect_binocular() for details.
        - saccade_inter_event_time: int; minimal time between two events in ms
        - saccade_min_duration: int; minimal duration of a saccade in ms
        - saccade_derivation_window_size: int; size of the derivation window in ms
        - saccade_lambda_noise_threshold: float; threshold for the noise in the derivation window

    :fixation keyword arguments:
        - fixation_detector_type: str; type of fixation detector to use:
            -- default: "velocity" (VelocityFixationDetector)
            -- None for no fixation detection
        - fixation_detect_by: str; controls how to detect fixations if using binocular data (default: 'both').
            see BaseDetector.detect_binocular() for details.
        - fixation_inter_event_time: int; minimal time between two events in ms
        - fixation_min_duration: int; minimal duration of a fixation in ms
        - fixation_velocity_threshold: float; threshold for the velocity in the derivation window

    :return:
    """
    sampling_rate = trial.sampling_rate
    _ts, x, y = trial.get_raw_gaze_data(eye=kwargs.pop("eye", "both"))

    is_blink = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                            detector_type=kwargs.pop("blink_detector_type", 'missing data'),  # change if we want to use blink detection
                            detect_by=kwargs.pop("blink_detect_by", 'either'),
                            inter_event_time=kwargs.pop("blink_inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME),
                            min_duration=kwargs.pop("blink_min_duration", cnfg.DEFAULT_BLINK_MINIMUM_DURATION),
                            missing_value=kwargs.pop("missing_value", cnfg.DEFAULT_MISSING_VALUE))

    is_saccade = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                              detector_type=kwargs.pop("saccade_detector_type", 'engbert'),  # change if we want to use saccade detection
                              detect_by=kwargs.pop("saccade_detect_by", 'both'),
                              inter_event_time=kwargs.pop("saccade_inter_event_time", cnfg.DEFAULT_INTER_EVENT_TIME),
                              min_duration=kwargs.pop("saccade_min_duration", cnfg.DEFAULT_SACCADE_MINIMUM_DURATION),
                              derivation_window_size=kwargs.pop("saccade_derivation_window_size",
                                                                DEFAULT_DERIVATION_WINDOW_SIZE),
                              lambda_noise_threshold=kwargs.pop("saccade_lambda_noise_threshold",
                                                                DEFAULT_LAMBDA_NOISE_THRESHOLD))

    dominant_eye = trial.subject.dominant_eye
    is_fixation = detect_event(x=x, y=y, sampling_rate=sampling_rate,
                               detector_type=kwargs.pop("fixation_detector_type", None),  # change if we want to use fixation detection
                               detect_by=kwargs.pop("fixation_detect_by", dominant_eye),
                               inter_event_time=kwargs.pop("fixation_inter_event_time",
                                                           cnfg.DEFAULT_INTER_EVENT_TIME),
                               min_duration=kwargs.pop("fixation_min_duration",
                                                       cnfg.DEFAULT_FIXATION_MINIMUM_DURATION),
                               velocity_threshold=kwargs.pop("fixation_velocity_threshold",
                                                             cnfg.DEFAULT_FIXATION_MAX_VELOCITY))

    is_blink, is_saccade, is_fixation = backfill_unidentified_samples(is_blink, is_saccade, is_fixation,
                                                                      fill_with=kwargs.pop("fill_with", "fixation"))
    return is_blink, is_saccade, is_fixation
