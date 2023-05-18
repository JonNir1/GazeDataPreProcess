# Example usage of the package

import os
import numpy as np
import pandas as pd

import constants as cnst
import experiment_config as cnfg

from DataParser.TobiiGazeDataParser import TobiiGazeDataParser
from Utils.ScreenMonitor import ScreenMonitor
from EventDetectors.scripts.detect_events import detect_all_events
from GazeEvents.scripts.extract_gaze_events import extract_events_to_dataframe

# create a screen monitor object
sm = ScreenMonitor.from_config()

# parse eye tracking data
path = r"Path/To/Your/Data"
parser = TobiiGazeDataParser(input_path=path, screen_monitor=sm)
sr = parser.sampling_rate
et_trials = parser.parse_and_split()  # returns a list of pd.DataFrames, one for each trial

# classify samples as blinks, saccades or fixations
tr = et_trials[0]
microseconds = tr[cnst.MICROSECONDS]
x, y = tr[cnst.LEFT_X], tr[cnst.LEFT_Y]
is_blink, is_saccade, is_fixation = detect_all_events(x, y, sampling_rate=sr,
                                                      blink_detector_type="missing_data",
                                                      saccade_detector_type="engbert",
                                                      stuff_with="fixation")

# extract events to a pandas DataFrame
blinks_summary = extract_events_to_dataframe(event_type="blink", timestamps=microseconds, is_event=is_blink,
                                             sampling_rate=sr)
saccades_summary = extract_events_to_dataframe(event_type="saccade", timestamps=microseconds, is_event=is_saccade,
                                               sampling_rate=sr, x=x, y=y)
fixations_summary = extract_events_to_dataframe(event_type="fixation", timestamps=microseconds, is_event=is_fixation,
                                                sampling_rate=sr, x=x, y=y)

# profit!
