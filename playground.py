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

pipline_config = {'save': True, 'skip_analysis': True, 'verbose': True}
subject1, subject1_analysis, failed_trials1 = ph.full_pipline(name="GalChen Demo", **pipline_config)
subject2, subject2_analysis, failed_trials2 = ph.full_pipline(name="Rotem Demo", **pipline_config)

# subject1 = ph.load_subject(subject_id=1, verbose=True)
# subject2 = ph.load_subject(subject_id=2, verbose=True)


##########################################
### PLAYGROUND  ##########################
##########################################

trial = subject1.get_all_trials()[0]
triggers = trial.get_triggers()
behavioral_data = trial.get_behavioral_data()
target_info = trial.get_targets()
gaze_events = trial.get_gaze_events()


