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

sm = ScreenMonitor.from_config()

trials = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=sm,
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

trial11 = trials[10]
trial11_raw_data = trial11._LWSTrial__behavioral_data._LWSBehavioralData__data
trial11_fixations = trial11.get_gaze_events(event_type=cnst.FIXATION)
trial11_fix1 = trial11_fixations[0]

end = time.time()
pp_duration = end - start
print(f"Finished preprocessing in: {pp_duration} seconds")
