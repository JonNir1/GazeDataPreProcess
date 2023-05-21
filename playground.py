import time
import numpy as np
import pandas as pd
import cv2

import constants as cnst
import experiment_config as cnfg

from Utils.ScreenMonitor import ScreenMonitor
import LWS.PreProcessing as pp


start = time.time()

sm = ScreenMonitor.from_config()
trials = pp.process_subject(subject_dir=r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\RawData\Rotem Demo",
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=sm,
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

trial1 = trials[0]

end = time.time()
print(f"Total time: {end - start}")

##########################################

for i, tr in enumerate(trials):
    ge = tr.get_gaze_events()
    fixs = list(filter(lambda e: e.event_type() == cnst.FIXATION, ge))
    centers = [f.center_of_mass for f in fixs]
    has_nans = any([np.isnan(c).any() for c in centers])
    if has_nans:
        print(f"Trial {i} has nans")
        break

tr22_data = tr._LWSTrial__behavioral_data._LWSBehavioralData__data
tr22_data[tr22_data["is_fixation"]]['right_y'].isna().sum()

# TODO: check why there are nans in the data
# TODO: use binocular data instead of monocular?

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
