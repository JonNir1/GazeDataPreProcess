import pandas as pd

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.scripts.check_proximity_to_target import check_target_proximity_for_gaze_data
from LWS.scripts.detect_events import detect_all_events
from LWS.scripts.extract_events import extract_all_events


def process_trial(trial: LWSTrial, sr: float, screen_monitor: ScreenMonitor = None, **kwargs):
    trial.set_is_processed(False)
    targets_data = trial.stimulus.get_target_data()
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()

    # process raw eye-tracking data
    is_blink, is_saccade, is_fixation = detect_all_events(trial, sr, **kwargs)
    is_close_to_target = check_target_proximity_for_gaze_data(trial, screen_monitor, cnfg.THRESHOLD_VISUAL_ANGLE)
    is_event_df = pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade,
                                'is_fixation': is_fixation, 'is_close_to_target': is_close_to_target})
    trial.behavioral_data.concat(is_event_df, deep_copy=False)  # add the new columns to the behavioral data
    pass
