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
from GazeEvents.scripts.extract_gaze_events import extract_events_to_dataframe

# create a screen monitor object
sm = ScreenMonitor.from_config()

# parse eye tracking data
et_data = parse_eye_tracker(et_type="tobii", et_path=r"Path/To/Your/Data", screen_monitor=sm, split_trials=False)

# classify samples as blinks, saccades or fixations
tr = et_data[0]
microseconds = tr[cnst.MICROSECONDS]
x, y = tr[cnst.LEFT_X], tr[cnst.LEFT_Y]
is_blink, is_saccade, is_fixation = detect_all_events(x, y, sampling_rate=sr,
                                                      blink_detector_type="missing_data",
                                                      saccade_detector_type="engbert",
                                                      stuff_with="fixation")

# extract events to a pandas DataFrame
sr = calculate_sampling_rate_from_microseconds(microseconds)
blinks_summary = extract_events_to_dataframe(event_type="blink", timestamps=microseconds, is_event=is_blink,
                                             sampling_rate=sr)
saccades_summary = extract_events_to_dataframe(event_type="saccade", timestamps=microseconds, is_event=is_saccade,
                                               sampling_rate=sr, x=x, y=y)
fixations_summary = extract_events_to_dataframe(event_type="fixation", timestamps=microseconds, is_event=is_fixation,
                                                sampling_rate=sr, x=x, y=y)

# profit!
