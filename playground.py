import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
from Config import experiment_config as cnfg
from LWS.DataModels.LWSSubject import LWSSubject
import Visualization.visualization_utils as visutils

# ##########################################
# ### PREPROCESSING GAZE DATA  #############
# ##########################################
#
# import LWS.PreProcessing as pp
#
# start = time.time()
#
# subject = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, 'Rotem Demo'),
#                              screen_monitor=cnfg.SCREEN_MONITOR,
#                              save_pickle=True,
#                              stuff_with='fixation',
#                              blink_detector_type='missing data',
#                              saccade_detector_type='engbert',
#                              drop_outlier_events=False)
#
# end = time.time()
# print(f"Finished preprocessing in: {(end - start):.2f} seconds")
#
# # delete irrelevant variables:
# del start, end

##########################################
###  LOAD DATA FROM PICKLE FILES  ########
##########################################

start = time.time()

subject = LWSSubject.from_pickle(os.path.join(cnfg.OUTPUT_DIR, "S002", "LWSSubject_002.pkl"))

end = time.time()
print(f"Finished loading in: {(end - start):.2f} seconds")
del start, end

# ##########################################
# ###  VISUALIZING DATA  ###################
# ##########################################
#
# from LWS.DataModels.LWSTrialVisualizer import LWSTrialVisualizer
#
# start = time.time()
#
# visualizer = LWSTrialVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution, output_directory=cnfg.OUTPUT_DIR)
#
# failed_trials = []
# for tr in subject.get_all_trials():
#     if tr.trial_num > 1:
#         break
#     try:
#         start_trial = time.time()
#         # visualizer.create_gaze_figure(trial=tr, savefig=True)
#         # visualizer.create_targets_figure(trial=tr, savefig=True)
#         visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=True, show_targets_color=(0, 0, 0))
#         visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=True, show_targets_color=(0, 0, 0))
#         # visualizer.create_video(trial=tr, output_directory=cnfg.OUTPUT_DIR)
#         end_trial = time.time()
#         print(f"\t{tr.__repr__()}:\t{(end_trial - start_trial):.2f} s")
#     except Exception as e:
#         failed_trials.append((tr, e))
#         print(f"Failed to visualize trial: {tr.__repr__()}")
#
# end = time.time()
# print(f"Finished visualization in: {(end - start):.2f} seconds")
#
# # delete irrelevant variables:
# del start, end, start_trial, end_trial, tr
# if len(failed_trials) == 0:
#     del failed_trials

##########################################
###  ANALYZING DATA  #####################
##########################################

import LWS.analysis_scripts.trial_summary as trsum
import LWS.analysis_scripts.events_summary as evsum
import Visualization.saccade_analysis as sacan
import LWS.analysis_scripts.fixation_analysis as fixan

start = time.time()

trials = subject.get_all_trials()

# all_blinks = [b for tr in trials for b in tr.get_gaze_events(cnst.BLINK)]
all_saccades = [s for tr in trials for s in tr.get_gaze_events(cnst.SACCADE)]
all_fixations = [f for tr in trials for f in tr.get_gaze_events(cnst.FIXATION)]
close_target_fixations = [f for f in all_fixations if f.visual_angle_to_target < cnfg.THRESHOLD_VISUAL_ANGLE]
mark_target_fixations = [f for f in all_fixations if f.is_mark_target_attempt]

# trial_summary = trsum.summarize_all_trials(trials)
# blink_summary = evsum.summarize_events(all_blinks)
# saccade_summary = evsum.summarize_events(all_saccades)
# fixation_summary = evsum.summarize_events(all_fixations)

sac_hists = sacan.saccade_histograms_figure(all_saccades, ignore_outliers=True)
all_fix_hists = fixan.fixation_histograms_figure(all_fixations, ignore_outliers=True,
                                                 title="All Fixations vs. Mark-Target Fixations")

visutils.show_figure(sac_hists)
visutils.show_figure(all_fix_hists)

end = time.time()
print(f"Finished analysis in: {(end - start):.2f} seconds")

del start, end


