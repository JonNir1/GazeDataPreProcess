# Example usage of the package

import os
import numpy as np
import pandas as pd

import constants as cnst
import experiment_config as cnfg

from Utils.ScreenMonitor import ScreenMonitor
from Utils.calculate_sampling_rate import calculate_sampling_rate_from_microseconds
from DataParser.scripts.parse_eye_tracker import parse_eye_tracker
from EventDetectors.scripts.detect_events import detect_all_events
from GazeEvents.scripts.gen_gaze_events import gen_gaze_events_summary

# create a screen monitor object
sm = ScreenMonitor.from_config()

# parse eye tracking data
et_data = parse_eye_tracker(et_type="tobii", et_path=r"Path/To/Your/Data",
                            screen_monitor=sm, split_trials=False,
                            additional_columns=cnfg.ADDITIONAL_COLUMNS)

# extract data from the first trial
tr = et_data[0]
microseconds = tr[cnst.MICROSECONDS].values
x_l, y_l = tr[cnst.LEFT_X].values, tr[cnst.LEFT_Y].values
x_r, y_r = tr[cnst.RIGHT_X].values, tr[cnst.RIGHT_Y].values
sr = calculate_sampling_rate_from_microseconds(microseconds)

# classify samples as blinks, saccades or fixations
is_blink, is_saccade, is_fixation = detect_all_events(x=np.vstack((x_l, x_r)), y=np.vstack((y_l, y_r)),
                                                      sampling_rate=sr, detect_by='both', fill_with="fixation",
                                                      blink_detector_type="missing_data",
                                                      saccade_detector_type="engbert")

# extract events to a pandas DataFrame
blinks_summary = gen_gaze_events_summary(event_type="blink", timestamps=microseconds, is_event=is_blink,)
saccades_summary = gen_gaze_events_summary(event_type="saccade", timestamps=microseconds, is_event=is_saccade, x=x_l, y=y_l)
fixations_summary = gen_gaze_events_summary(event_type="fixation", timestamps=microseconds, is_event=is_fixation, x=x_l, y=y_l)

# profit!
