import os
import time
import numpy as np
import pandas as pd
import cv2

import constants as cnst
import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
import LWS.PreProcessing as pp

start = time.time()

trials = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=ScreenMonitor.from_config(),
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

trial11 = trials[10]
trial11_raw_data = trial11._LWSTrial__behavioral_data._LWSBehavioralData__data
trial11_fixations = trial11.get_gaze_events(event_type=cnst.FIXATION)
trial11_fix1 = trial11_fixations[0]

end = time.time()
print(f"Finished preprocessing in: {end - start} seconds")
del start
del end

##########################################

from typing import List


def split_samples_between_events(is_event: np.ndarray) -> List[np.ndarray]:
    # returns a list of arrays, each array contains the indices of the samples that belong to the same event
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    return different_event_idxs

timestamps, x, y = trial11.get_raw_gaze_coordinates()
trial11_triggers = trial11.get_behavioral_data().get(cnst.TRIGGER).values
is_fixation = trial11.get_behavioral_data().get("is_fixation").values
separate_event_idxs = split_samples_between_events(is_fixation)

trigs = []
for i, idxs in enumerate(separate_event_idxs):
    ts = timestamps[idxs]
    trgs = trial11_triggers[idxs]
    trigs.append({ts[i]: int(trgs[i]) for i in range(len(idxs)) if not np.isnan(trgs[i])})


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
