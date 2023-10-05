import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import traceback
# from typing import Optional, Tuple, List, Union, Dict

import constants as cnst
import playground_helpers as ph
from Config import experiment_config as cnfg
import Visualization.visualization_utils as visutils

##########################################
### RUN PIPELINE / LOAD DATA  ############
##########################################

# subjects: "GalChen Demo" (001), "Rotem Demo" (002), "Netta Demo" (003)
pipline_config = {'save': True, 'include_subject_figures': True,
                  'include_trial_figures': False, 'include_trial_videos': False, 'verbose': True}

subject1, subject1_figures, failed_trials1 = ph.full_pipline(name_or_id="GalChen Demo", **pipline_config)

# subject2, subject2_figures, failed_trials2 = ph.full_pipline(name_or_id="Rotem Demo", **pipline_config)
#
# subject3, subject3_figures, failed_trials3 = ph.full_pipline(name_or_id="Netta Demo", **pipline_config)
#
# del pipline_config, subject1_figures, failed_trials1, subject2_figures, failed_trials2, subject3_figures, failed_trials3

# subject1 = ph.load_or_preprocess_subject(name_or_id=1, verbose=True)
# subject2 = ph.load_or_preprocess_subject(name_or_id=2, verbose=True)
# subject3 = ph.load_or_preprocess_subject(name_or_id=3, verbose=True)


##########################################
### RUN PIPELINE for Real Subjects  ######
##########################################

# pipline_config = {'save': True, 'include_subject_dfs': True, 'include_subject_figures': True,
#                   'include_trial_figures': True, 'include_trial_videos': True, 'verbose': True}
#
# subject_names = [d for d in os.listdir(cnfg.RAW_DATA_DIR) if d.startswith("v5-")]
# subjects = {}
#
# for s in subject_names:
#     try:
#         subj, subj_dfs, subj_figures, subj_failed_trials = ph.full_pipline(name=s, **pipline_config)
#         subjects[s] = (True, subj, subj_dfs, subj_figures, subj_failed_trials)
#     except Exception as e:
#         trace = traceback.format_exc()
#         subjects[s] = (False, e, trace)
#     break
#
# del pipline_config, subject_names

##########################################
### PLAYGROUND  ##########################
##########################################

from LWS.SubjectAnalysis.search_analysis.target_identification import get_target_identification_data

subject1 = ph.load_or_preprocess_subject(name_or_id=1, verbose=True)
target_identifications = pd.DataFrame({tr: get_target_identification_data(tr, 2)['distance_identified']
                                       for tr in subject1.get_trials()}).T
num_targets_per_trial = target_identifications.isna().sum(axis=1)
num_unidentified_per_trial = np.isinf(target_identifications).sum(axis=1)



