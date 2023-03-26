import numpy as np
import pandas as pd

import constants as cnst
import experiment_config as conf

from scripts.calculate_sampling_rate import calculate_sampling_rate_for_tobii
from scripts.parse_and_merge import parse_tobii_gaze_and_triggers
from scripts.detect_events import detect_all_events

sr = calculate_sampling_rate_for_tobii(r"C:\Users\jonathanni\Desktop\GazeData.txt")
trial_dfs = parse_tobii_gaze_and_triggers(r"C:\Users\jonathanni\Desktop\GazeData.txt",
                                          r"C:\Users\jonathanni\Desktop\TriggerLog.txt", start_trigger=254,
                                          end_trigger=255)
t2 = trial_dfs[2]

is_blink, is_saccade, is_fixation = detect_all_events(x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values,
                                                     sampling_rate=sr, inter_event_time=5,
                                                     blink_detector_type='missing data',
                                                     saccade_detector_type='engbert', saccade_min_duration=conf.SACCADE_MINIMUM_DURATION)


