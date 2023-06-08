import os
import pandas as pd
from typing import List

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial

from LWS.scripts.read_subject import read_subject_trials
from LWS.scripts.distance_to_targets import calculate_angular_distance_for_gaze_data
from LWS.scripts.detect_events import detect_all_events
from LWS.scripts.extract_events import extract_all_events


def process_subject(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR, **kwargs) -> List[LWSTrial]:
    """
    For a given subject directory, extracts the subject-info, gaze-data and trigger-log files, and uses those to create
    the LWSTrial objects of that subject. Then, each trial is processed so that we detect blinks, saccades and fixations
    and add the processed data to the trial object.

    :param subject_dir: directory containing the subject's data files.
    :param stimuli_dir: directory containing the stimuli files.

    keyword arguments:
        - screen_monitor: The screen monitor used to display the stimuli.

    gaze detection keyword arguments:
        - stuff_with: either "saccade", "fixation" or None. Controls how to fill unidentified samples.
        - blink_detector_type: The type of blink detector to use. If None, no blink detection is performed.
        - saccade_detector_type: The type of saccade detector to use. If None, no saccade detection is performed.
        - fixation_detector_type: The type of fixation detector to use. If None, no fixation detection is performed.
        - drop_outlier_events: If True, drops events that are considered outliers. If False, keeps all events.

    blink keyword arguments:
        - blink_inter_event_time: minimal time between two events in ms;                                default: 5 ms
        - blink_min_duration: minimal duration of a blink in ms;                                        default: 50 ms
        - missing_value: default value indicating missing data, used by MissingDataBlinkDetector;       default: np.nan

    saccade keyword arguments:
        - saccade_inter_event_time: minimal time between two events in ms;                              default: 5 ms
        - saccade_min_duration: minimal duration of a blink in ms;                                      default: 5 ms
        - derivation_window_size: window size for derivation in ms;                                     default: 3 ms
        - lambda_noise_threshold: threshold for lambda noise, used by EngbertSaccadeDetector;           default: 5

    fixation keyword arguments:
        - fixation_inter_event_time: minimal time between two events in ms;                             default: 5 ms
        - fixation_min_duration: minimal duration of a blink in ms;                                     default: 55 ms
        - velocity_threshold: maximal velocity allowed within a fixation, used by IVTFixationDetector;  default: 30 deg/s

    :return: A list of LWSTrial objects, one for each trial of the subject, processed and ready to be analyzed.
    """
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")

    kwargs["screen_monitor"] = kwargs.get("screen_monitor", None) or ScreenMonitor.from_config()
    trials = read_subject_trials(subject_dir, stimuli_dir, **kwargs)
    for _i, trial in enumerate(trials):
        process_trial(trial, **kwargs)
    return trials


def process_trial(trial: LWSTrial, **kwargs):
    """
    Processes the given trial and adds the processed data to the trial object.

    keyword arguments:
        - screen_monitor: The screen monitor used to display the stimuli.

    gaze detection keyword arguments:
        - stuff_with: either "saccade", "fixation" or None. Controls how to fill unidentified samples.
        - blink_detector_type: The type of blink detector to use. If None, no blink detection is performed.
        - saccade_detector_type: The type of saccade detector to use. If None, no saccade detection is performed.
        - fixation_detector_type: The type of fixation detector to use. If None, no fixation detection is performed.
        - drop_outlier_events: If True, drops events that are considered outliers. If False, keeps all events.

    blink keyword arguments:
        - blink_inter_event_time: minimal time between two events in ms;                                default: 5 ms
        - blink_min_duration: minimal duration of a blink in ms;                                        default: 50 ms
        - missing_value: default value indicating missing data, used by MissingDataBlinkDetector;       default: np.nan

    saccade keyword arguments:
        - saccade_inter_event_time: minimal time between two events in ms;                              default: 5 ms
        - saccade_min_duration: minimal duration of a blink in ms;                                      default: 5 ms
        - derivation_window_size: window size for derivation in ms;                                     default: 3 ms
        - lambda_noise_threshold: threshold for lambda noise, used by EngbertSaccadeDetector;           default: 5

    fixation keyword arguments:
        - fixation_inter_event_time: minimal time between two events in ms;                             default: 5 ms
        - fixation_min_duration: minimal duration of a blink in ms;                                     default: 55 ms
        - velocity_threshold: maximal velocity allowed within a fixation, used by IVTFixationDetector;  default: 30 deg/s
    """
    trial.is_processed = False
    sm = kwargs.pop('screen_monitor', None) or ScreenMonitor.from_config()
    bd = trial.get_behavioral_data()

    # process raw eye-tracking data
    is_blink, is_saccade, is_fixation = detect_all_events(trial, **kwargs)
    target_distance = calculate_angular_distance_for_gaze_data(trial, sm=sm)
    is_event_df = pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade,
                                'is_fixation': is_fixation, 'target_distance': target_distance}, index=bd.index)

    # add the new columns to the behavioral data:
    new_behavioral_data = bd.concat(is_event_df)
    trial.set_behavioral_data(new_behavioral_data)

    # process gaze events
    drop_outlier_events = kwargs.pop('drop_outlier_events', False)
    events = extract_all_events(trial, sm, drop_outliers=drop_outlier_events)
    trial.set_gaze_events(events)

    trial.is_processed = True
