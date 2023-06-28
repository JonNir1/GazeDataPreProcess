import os
import time
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
from Config import experiment_config as cnfg
from LWS.DataModels.LWSSubject import LWSSubject

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

##########################################
###  VISUALIZING DATA  ###################
##########################################

from LWS.DataModels.LWSTrialVisualizer import LWSTrialVisualizer

start = time.time()

visualizer = LWSTrialVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution, output_directory=cnfg.OUTPUT_DIR)

failed_trials = []
for i in range(subject.num_trials):
    if i > 0:
        break
    tr = subject.get_trial(i+1)  # trial numbers start from 1
    try:
        start_trial = time.time()
        visualizer.create_gaze_figure(trial=tr, savefig=True)
        visualizer.create_targets_figure(trial=tr, savefig=True)
        visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=True, show_targets_color=(0, 0, 0))
        visualizer.create_heatmap(trial=tr, savefig=True, fixation_only=False, show_targets_color=(0, 0, 0))
        visualizer.create_video(trial=tr, output_directory=cnfg.OUTPUT_DIR)
        end_trial = time.time()
        print(f"\t{tr.__repr__()}:\t{(end_trial - start_trial):.2f} s")
    except Exception as e:
        failed_trials.append((tr, e))
        print(f"Failed to visualize trial: {tr.__repr__()}")

end = time.time()
print(f"Finished visualization in: {(end - start):.2f} seconds")

# delete irrelevant variables:
del start, end, start_trial, end_trial, tr

##########################################
###  ANALYZING DATA  #####################
##########################################

import LWS.analysis_scripts.trial_summary as trsum
import LWS.analysis_scripts.events_summary as evsum

start = time.time()

trials = subject.get_all_trials()

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


