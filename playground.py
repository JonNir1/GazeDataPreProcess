import numpy as np
import pandas as pd

import constants as cnst
import EventDetectors.utils as u

from DataParser.TobiiGazeDataParser import TobiiGazeDataParser
from EventDetectors.MonocularBlinkDetector import MonocularBlinkDetector
from EventDetectors.BinocularBlinkDetector import BinocularBlinkDetector


path = r"C:\Users\jonathanni\Desktop\b.txt"
tobii_parser = TobiiGazeDataParser(path)
sr = tobii_parser.sampling_rate

trial_dfs = tobii_parser.parse_and_split()
trial2 = trial_dfs[1]


binoc_blink_detector = BinocularBlinkDetector(criterion="and")
binoc_blink_detector.set_sampling_rate(sr)
is_blink_binoc = binoc_blink_detector.detect(trial2[cnst.LEFT_X].values, trial2[cnst.LEFT_Y].values,
                                             trial2[cnst.RIGHT_X].values, trial2[cnst.RIGHT_Y].values)


left_x = trial2[cnst.LEFT_X].values
left_y = trial2[cnst.LEFT_Y].values

