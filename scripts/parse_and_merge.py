import numpy as np
import pandas as pd

import constants as cnst
from DataParser.TobiiGazeDataParser import TobiiGazeDataParser
from DataParser.TriggerLogParser import TriggerLogParser


def parse_and_merge_tobii_triggers(tobii_path, trigger_path,
                                   start_trigger: int = 254, end_trigger: int = 255):
    tobii_parser = TobiiGazeDataParser(tobii_path)
    trigger_parser = TriggerLogParser(trigger_path, start_trigger=start_trigger, end_trigger=end_trigger)

    gaze_dfs = tobii_parser.parse_and_split()
    trigger_dfs = trigger_parser.parse_and_split()
    merged_dfs = []
    for gaze_df, trigger_df in zip(gaze_dfs, trigger_dfs):
        merged_df = pd.merge_asof(gaze_df, trigger_df.drop(columns=cnst.TRIAL),
                                  on=cnst.MILLISECONDS, direction='backward')
        merged_df[cnst.TRIGGER][merged_df[cnst.TRIGGER].diff() == 0] = np.nan  # keep only the first instance of a trigger
        merged_dfs.append(merged_df)
    return merged_dfs
