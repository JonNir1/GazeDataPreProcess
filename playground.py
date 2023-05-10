import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import constants as cnst
import experiment_config as cnfg

from LWS.scripts.read_subject import read_subject
from Utils.ScreenMonitor import ScreenMonitor

monitor = ScreenMonitor.from_config()
sr, trials = read_subject(subject_dir=r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\RawData\Rotem Demo",
                          screen_monitor=monitor)
trial1 = trials[0]



from DataParser.scripts.parse_tobii_gaze_and_triggers import parse_tobii_gaze_and_triggers
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



# create a video of eye movements
screen_w, screen_h = cnfg.SCREEN_RESOLUTION
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, round(sr), (screen_w, screen_h))

for i, row in t2_with_events.iterrows():
    x = row[cnst.LEFT_X]
    y = row[cnst.LEFT_Y]

    # create a blank image and add circle around the gaze point
    img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if not pd.isna(x) and not pd.isna(y):
        cv2.circle(img, (round(x), round(y)),
                   10, (0, 0, 255), -1)
    video.write(img)

# release the video
video.release()
