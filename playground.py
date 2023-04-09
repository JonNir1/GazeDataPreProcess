import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import cv2

import constants as cnst
import experiment_config as cnfg
from DataParser.scripts.parse_and_merge import parse_tobii_gaze_and_triggers
from EventDetectors.scripts.detect_events import detect_all_events
from GazeEvents.scripts.extract_gaze_events import extract_events_to_dataframe

sr, trial_dfs = parse_tobii_gaze_and_triggers(r"C:\Users\jonathanni\Desktop\GazeData.txt",
                                              r"C:\Users\jonathanni\Desktop\TriggerLog.txt", start_trigger=254,
                                              end_trigger=255)
t2 = trial_dfs[2]

is_blink, is_saccade, is_fixation = detect_all_events(x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values,
                                                      sampling_rate=sr,
                                                      stuff_with='fixation',
                                                      blink_detector_type='missing data',
                                                      saccade_detector_type='engbert')
t2_with_events = pd.concat([t2, pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade,
                                              'is_fixation': is_fixation})], axis=1)

fe = extract_events_to_dataframe(event_type='fixation', timestamps=t2[cnst.MICROSECONDS].values / 1000,
                                 is_event=is_fixation, sampling_rate=sr,
                                 x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values)
