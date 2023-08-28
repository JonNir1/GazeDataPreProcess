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
# pipline_config = {'save': True, 'include_subject_dfs': True, 'include_subject_figures': True,
#                   'include_trial_figures': True, 'include_trial_videos': True, 'verbose': True}
#
# subject1, subject1_analysis, failed_trials1 = ph.full_pipline(name="GalChen Demo", **pipline_config)
# del subject1_analysis, failed_trials1
#
# subject2, subject2_analysis, failed_trials2 = ph.full_pipline(name="Rotem Demo", **pipline_config)
# del subject2_analysis, failed_trials2
#
# del pipline_config

subject1 = ph.load_subject(subject_id=1, verbose=True)
subject2 = ph.load_subject(subject_id=2, verbose=True)


##########################################
### PLAYGROUND  ##########################
##########################################

