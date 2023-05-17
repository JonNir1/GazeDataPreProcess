import time
import numpy as np
import pandas as pd
import cv2

import constants as cnst
import experiment_config as cnfg

from Utils.ScreenMonitor import ScreenMonitor
from LWS.scripts.read_subject import read_subject
from LWS.scripts.detect_events import detect_all_events
from LWS.scripts.extract_events import extract_all_events, extract_event


start = time.time()

sm = ScreenMonitor.from_config()
sr, trials = read_subject(subject_dir=r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\RawData\Rotem Demo",
                          screen_monitor=sm)
trial1 = trials[0]
is_blink, is_saccade, is_fixation = detect_all_events(trial=trial1, sampling_rate=sr,
                                                      stuff_with='fixation',
                                                      blink_detector_type='missing data',
                                                      saccade_detector_type='engbert')
trial1.behavioral_data.concat(pd.DataFrame({'is_blink': is_blink, 'is_saccade': is_saccade, 'is_fixation': is_fixation}))
# t_df = trial1.behavioral_data._LWSBehavioralData__data

fix_events_list = extract_event(trial=trial1, event_type='fixation', sampling_rate=sr, screen_monitor=sm)
all_events_list = extract_all_events(trial=trial1, sampling_rate=sr, screen_monitor=sm, drop_outliers=False)


end = time.time()
print(f"Total time: {end - start}")

##########################################

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
