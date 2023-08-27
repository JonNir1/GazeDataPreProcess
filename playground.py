import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
import playground_helpers as ph
from Config import experiment_config as cnfg
import Visualization.visualization_utils as visutils

##########################################
### RUN PIPELINE / LOAD DATA  ############
##########################################

# subjects: "GalChen Demo" (001), "Rotem Demo" (002)
pipline_config = {'save': True, 'run_analysis': True, 'run_visualization': True, 'verbose': True}

subject1, subject1_analysis, failed_trials1 = ph.full_pipline(name="GalChen Demo", **pipline_config)
del subject1_analysis, failed_trials1

subject2, subject2_analysis, failed_trials2 = ph.full_pipline(name="Rotem Demo", **pipline_config)
del subject2_analysis, failed_trials2

del pipline_config

# subject1 = ph.load_subject(subject_id=1, verbose=True)
# subject2 = ph.load_subject(subject_id=2, verbose=True)


##########################################
### PLAYGROUND  ##########################
##########################################

start = time.time()

from LWS.subject_analysis.lws_instances import calculate_lws_rate as calc_lws_rate

prox_thresholds = np.arange(0.1, 7.1, 0.1)
time_difference_thresholds = np.arange(0, 251, 10)
trials = subject1.get_all_trials()
lws_rates = np.full((len(trials), len(prox_thresholds), len(time_difference_thresholds)), np.nan)

for i, trial in enumerate(trials):
    trial_start = time.time()
    print(f"\t{str(trial)}")
    for j, prox_threshold in enumerate(prox_thresholds):
        for k, time_threshold in enumerate(time_difference_thresholds):
            lws_rates[i, j, k] = calc_lws_rate(trial,
                                               proximity_threshold=prox_threshold,
                                               time_difference_threshold=time_threshold)
    print(f"\t\ttook {time.time() - trial_start:.2f} seconds\n")

elapsed = time.time() - start
print(f"Finished in {elapsed:.2f} seconds")
del start, elapsed, trial_start, i, j, k, trial, prox_threshold, time_threshold

###################

from Config.ExperimentTriggerEnum import ExperimentTriggerEnum
from LWS.subject_analysis.keyboard_inputs import trigger_counts_for_block_position as triggers_in_block
target_marking = triggers_in_block(subject1, triggers=ExperimentTriggerEnum.MARK_TARGET_SUCCESSFUL)
target_confirmation = triggers_in_block(subject1, triggers=ExperimentTriggerEnum.CONFIRM_TARGET_SUCCESSFUL)
target_rejection = triggers_in_block(subject1, triggers=ExperimentTriggerEnum.REJECT_TARGET_SUCCESSFUL)
counts_df = pd.DataFrame([target_marking, target_confirmation, target_rejection]).T.rename(columns={0: "mark", 1: "confirm", 2: "reject"})
del target_marking, target_confirmation, target_rejection



