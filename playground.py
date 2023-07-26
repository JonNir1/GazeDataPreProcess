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

subject, subject_analysis, failed_trials = ph.full_pipline(name="GalChen Demo", save=True, verbose=True)

##########################################
### PLAYGROUND  ##########################
##########################################




