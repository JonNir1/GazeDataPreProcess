import numpy as np

import constants as cnst

from DataParser.scripts.parse_and_merge import parse_tobii_gaze_and_triggers
from EventDetectors.scripts.detect_events import detect_all_events
from GazeEvents.FixationEvent import FixationEvent

sr, trial_dfs = parse_tobii_gaze_and_triggers(r"C:\Users\jonathanni\Desktop\GazeData.txt",
                                              r"C:\Users\jonathanni\Desktop\TriggerLog.txt", start_trigger=254,
                                              end_trigger=255)
t2 = trial_dfs[2]

is_blink, is_saccade, is_fixation = detect_all_events(x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values,
                                                      sampling_rate=sr,
                                                      stuff_with='fixation',
                                                      blink_detector_type='missing data',
                                                      saccade_detector_type='engbert')

fe = FixationEvent.extract_fixation_events(timestamps=t2[cnst.MICROSECONDS].values / 1000,
                                           is_fixation=is_fixation, x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values,
                                           sampling_rate=sr)
