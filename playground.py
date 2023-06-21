import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSTrialVisualizer import LWSTrialVisualizer

##########################################

start = time.time()

sm = ScreenMonitor.from_config()
trial = LWSTrial.from_pickle(os.path.join(cnfg.OUTPUT_DIR, "S002", "trials", "LWSTrial_S2_T1.pkl"))
visualizer = LWSTrialVisualizer(screen_resolution=sm.resolution, output_directory=cnfg.OUTPUT_DIR)

fig = visualizer.create_gaze_plot(trial=trial, show=False)
fig.show()

end = time.time()
print(f"Finished loading in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start
del end

##########################################

import LWS.PreProcessing as pp
# from LWS.DataModels.LWSFixationEvent import LWSFixationEvent

start = time.time()

trials = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=sm,
                            save_pickle=True,
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

# for i, tr in enumerate(trials):
#     fixations: List[LWSFixationEvent] = tr.get_gaze_events(cnst.FIXATION)
#     fixations_target_distances = np.array([f.visual_angle_to_target for f in fixations])
#     if any(fixations_target_distances <= 1.5):
#         break

end = time.time()
print(f"Finished preprocessing in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start
del end

##########################################

for i, tr in enumerate(trials):
    try:
        start_trial = time.time()
        visualizer.create_video(trial=tr, output_directory=cnfg.OUTPUT_DIR)
        end_trial = time.time()
        print(f"\t{str(tr)}:\t{(end_trial - start_trial):.2f} s")
    except Exception as e:
        print(f"\nFailed to visualize {str(tr)}: {{e}}")
        print("\n")

end = time.time()
print(f"Finished visualization in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start
del end
del start_trial
del end_trial
del i
del tr

