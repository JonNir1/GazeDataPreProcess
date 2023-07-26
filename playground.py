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
### RUN THE PIPELINE  ####################
##########################################

# subjects: "GalChen Demo" (001), "Rotem Demo" (002)
# subject = ph.process_subject(name="GalChen Demo", save=True, verbose=True)

subject = ph.load_subject(subject_id=1, verbose=True)

subject_analysis = ph.analyze_subject(subject, save=True, verbose=True)
trial_summary, saccade_distributions, fixation_distributions, fixation_dynamics, fixation_proximity_comparison = subject_analysis
del subject_analysis

ph.visualize_all_trials(subject, save=True, verbose=True)

##########################################
### PLAYGROUND  ##########################
##########################################




