import pandas as pd

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.scripts.detect_events import detect_all_events
from LWS.scripts.check_proximity_to_target import check_proximity_to_target


def process_trial(trial: LWSTrial, sr: float, screen_monitor: ScreenMonitor = None, **kwargs):
    trial.set_is_processed(False)
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()
    is_blink, is_saccade, is_fixation = detect_all_events(trial, sr, **kwargs)
    is_event_df = pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade, 'is_fixation': is_fixation})
    trial.behavioral_data.concat(is_event_df)
    pass
