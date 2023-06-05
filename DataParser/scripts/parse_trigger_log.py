import os
import pandas as pd
from typing import List, Union

import experiment_config as cnfg
from DataParser.EPrimeTriggerLogParser import EPrimeTriggerLogParser


def parse_trigger_log(path: str,
                      start_trigger: int = cnfg.START_RECORDING_TRIGGER,
                      end_trigger: int = cnfg.END_RECORDING_TRIGGER,
                      split_trials: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Parse trigger log file and return a dataframe with the parsed data, or a list of dataframes if split_trials is True.

    :raise FileNotFoundError: if the file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    parser = EPrimeTriggerLogParser(start_trigger=start_trigger, end_trigger=end_trigger)
    if split_trials:
        return parser.parse_and_split(path)
    return parser.parse(path)
