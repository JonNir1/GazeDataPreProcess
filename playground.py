import os
import time
import numpy as np
import pandas as pd
import cv2

import constants as cnst
import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
import LWS.PreProcessing as pp
from LWS.DataModels.LWSVisualizer import LWSVisualizer

start = time.time()

sm = ScreenMonitor.from_config()

trials = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=sm,
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

end = time.time()
print(f"Finished preprocessing in: {(end - start):.2f} seconds")


start = time.time()

visualizer = LWSVisualizer(screen_monitor=sm)

for i, tr in enumerate(trials):
    try:
        start_trial = time.time()
        visualizer.visualize(trial=tr, output_directory=cnfg.OUTPUT_DIR, show=False)
        end_trial = time.time()
        print(f"\tTrial {i}:\t{(end_trial - start_trial):.2f} s")
    except Exception as e:
        print(f"\nFailed to visualize trial {i}")
        print(e)
        print("\n")

end = time.time()
print(f"Finished visualization in: {(end - start):.2f} seconds")

##########################################
# keep only relevant variables:

del start
del end
del start_trial
del end_trial
del i
del tr

# trial11 = trials[10]
# trial11_raw_data = trial11._LWSTrial__behavioral_data._LWSBehavioralData__data
# trial11_fixations = trial11.get_gaze_events(event_type=cnst.FIXATION)
# trial11_fix1 = trial11_fixations[0]

