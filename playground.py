import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial

sm = ScreenMonitor.from_config()

##########################################
###  LOADING DATA FROM PICKLE FILES  #####
##########################################

start = time.time()

trials = [LWSTrial.from_pickle(os.path.join(cnfg.OUTPUT_DIR, "S002", "trials", f"LWSTrial_S2_T{i+1}.pkl")) for i in range(60)]

end = time.time()
print(f"Finished loading in: {(end - start):.2f} seconds")
del start, end

##########################################
###  VISUALIZING DATA  ###################
##########################################

from LWS.DataModels.LWSTrialVisualizer import LWSTrialVisualizer

start = time.time()

visualizer = LWSTrialVisualizer(screen_resolution=sm.resolution, output_directory=cnfg.OUTPUT_DIR)

for tr in trials:
    start_trial = time.time()
    visualizer.create_gaze_figure(trial=tr, savefig=True)
    visualizer.create_targets_figure(trial=tr, savefig=True)
    # visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=True, show_targets_color=(0, 0, 0))
    # visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=False, show_targets_color=(0, 0, 0))
    # visualizer.create_video(trial=tr, output_directory=cnfg.OUTPUT_DIR)
    end_trial = time.time()
    print(f"\t{tr.__repr__()}:\t{(end_trial - start_trial):.2f} s")
    if tr.trial_num > -1:
        break

end = time.time()
print(f"Finished visualization in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start, end, tr

##########################################
###  ANALYZING DATA  #####################
##########################################

import LWS.analysis_scripts.trial_summary as trsum
import LWS.analysis_scripts.events_summary as evsum

start = time.time()

trial_summary = trsum.summarize_all_trials(trials)

all_blinks = [b for tr in trials for b in tr.get_gaze_events(cnst.BLINK)]
blink_summary = evsum.summarize_events(all_blinks)

all_saccades = [s for tr in trials for s in tr.get_gaze_events(cnst.SACCADE)]
saccade_summary = evsum.summarize_events(all_saccades)

all_fixations = [f for tr in trials for f in tr.get_gaze_events(cnst.FIXATION)]
fixation_summary = evsum.summarize_events(all_fixations)

end = time.time()
print(f"Finished analysis in: {(end - start):.2f} seconds")

del start, end

##########################################
### PREPROCESSING GAZE DATA  #############
##########################################

import LWS.PreProcessing as pp

start = time.time()

sm = ScreenMonitor.from_config()

trials = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
                            stimuli_dir=cnfg.STIMULI_DIR,
                            screen_monitor=sm,
                            save_pickle=True,
                            stuff_with='fixation',
                            blink_detector_type='missing data',
                            saccade_detector_type='engbert',
                            drop_outlier_events=False)

end = time.time()
print(f"Finished preprocessing in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start, end, start_trial, end_trial

