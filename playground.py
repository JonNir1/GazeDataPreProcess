import time
import numpy as np
import pandas as pd
import cv2

import constants as cnst
import experiment_config as cnfg

from Utils.ScreenMonitor import ScreenMonitor
from LWS.scripts.read_subject import read_subject_trials
from LWS.scripts.process_trial import process_trial


start = time.time()

sm = ScreenMonitor.from_config()
trials = read_subject_trials(subject_dir=r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\RawData\Rotem Demo",
                                 stimuli_dir=cnfg.STIMULI_DIR,
                                 screen_monitor=sm)
trial1 = trials[0]

process_trial(trial1, screen_monitor=sm, stuff_with='fixation',
              blink_detector_type='missing data',
              saccade_detector_type='engbert',
              drop_outlier_events=False)

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
