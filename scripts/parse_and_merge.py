import numpy as np
import pandas as pd

import constants as cnst
from DataParser.TobiiGazeDataParser import TobiiGazeDataParser
from DataParser.TriggerLogParser import TriggerLogParser


def parse_tobii_gaze_and_triggers(gaze_path, trigger_path,
                                  start_trigger: int = 254, end_trigger: int = 255):
    """
    Parse tobii gaze data and trigger log and merge them into a single dataframe for each trial.
    :param gaze_path: path to the tobii gaze data file
    :param trigger_path: path to the trigger-log file
    :param start_trigger: trigger indicating start of a trial
    :param end_trigger: trigger indicating end of a trial

    :return: list of dataframes, one for each trial
    """
    tobii_parser = TobiiGazeDataParser(gaze_path)
    trigger_parser = TriggerLogParser(trigger_path, start_trigger=start_trigger, end_trigger=end_trigger)

    gaze_dfs = tobii_parser.parse_and_split()
    trigger_dfs = trigger_parser.parse_and_split()
    merged_dfs = []
    for gaze_df, trigger_df in zip(gaze_dfs, trigger_dfs):
        merged_df = pd.merge_asof(gaze_df, trigger_df.drop(columns=cnst.TRIAL),
                                  on=cnst.MILLISECONDS, direction='backward')
        same_trigger = merged_df[cnst.TRIGGER].diff() == 0
        merged_df.loc[same_trigger, cnst.TRIGGER] = np.nan  # keep only the first instance of a trigger
        merged_dfs.append(merged_df)
    return merged_dfs
