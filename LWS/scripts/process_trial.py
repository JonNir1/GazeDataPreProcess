import pandas as pd

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.scripts.distance_to_targets import calculate_angular_distance_for_gaze_data
from LWS.scripts.detect_events import detect_all_events
from LWS.scripts.extract_events import extract_all_events


def process_trial(trial: LWSTrial, sr: float, screen_monitor: ScreenMonitor = None, **kwargs):
    """
    Processes the given trial and adds the processed data to the trial object.

    TODO: add docstring

    :param trial:
    :param sr:
    :param screen_monitor:
    :param kwargs:
    :return:
    """
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()

    trial.set_is_processed(False)

    # process raw eye-tracking data
    is_blink, is_saccade, is_fixation = detect_all_events(trial, sr, **kwargs)
    target_distance = calculate_angular_distance_for_gaze_data(trial, screen_monitor)
    is_event_df = pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade,
                                'is_fixation': is_fixation, 'target_distance': target_distance})
    trial.get_behavioral_data().concat(is_event_df, deep_copy=False)  # add the new columns to the behavioral data

    # process gaze events
    drop_outlier_events = kwargs.pop('drop_outlier_events', False)
    events = extract_all_events(trial, screen_monitor, sr, drop_outliers=drop_outlier_events)
    trial.set_gaze_events(events)

    trial.set_is_processed(True)
