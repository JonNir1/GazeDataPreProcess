import os
import numpy as np
import pandas as pd
import scipy as sp

import constants as cnst

from LWSStimuli.ImageArray import ImageArray

ia1 = ImageArray.from_file(r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\Stimuli\generated_stim1\bw\image_1.mat")



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

fe = extract_events_to_dataframe(event_type='fixation', timestamps=t2[cnst.MICROSECONDS].values / 1000,
                                 is_event=is_fixation, sampling_rate=sr,
                                 x=t2[cnst.LEFT_X].values, y=t2[cnst.LEFT_Y].values)
