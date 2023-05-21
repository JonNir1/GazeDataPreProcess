import os
import pandas as pd
from typing import Optional, List, Union

from Utils.ScreenMonitor import ScreenMonitor
from DataParser.BaseEyeTrackingParser import BaseEyeTrackingParser


def parse_eye_tracker(et_type: str, et_path: str,
                      split_trials: bool = True, **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Parses the eye tracking data from the given path, using the given eye tracker type.
    If `split_trials` is True, the data will be split into trials.

    :param et_type: name of the eye tracker
    :param et_path: path to the eye tracking data
    :param split_trials: if True, the data will be split into trials

    :keyword screen_monitor: the screen monitor used for the experiment
    :keyword additional_columns: additional columns to keep from the eye tracking data
    :keyword output_path: path to save the parsed data

    :return: the parsed data, either as a single DataFrame or as a list of DataFrames (one for each trial)

    :raise FileNotFoundError: if the given path does not exist
    :raise ValueError: if the given eye tracker type is unknown
    """
    if not os.path.exists(et_path):
        raise FileNotFoundError(f'File not found: {et_path}')

    # extract keyword arguments:
    sm = kwargs.get('screen_monitor', None)
    additional_columns = kwargs.get('additional_columns', None)
    output_path = kwargs.get('output_path', None)

    # get the parser and parse the data:
    et_parser = _get_eye_tracker_parser(et_type, additional_columns, sm)
    if split_trials:
        return et_parser.parse_and_split(et_path, output_path)
    return et_parser.parse(et_path, output_path)


def _get_eye_tracker_parser(et_type: str,
                            additional_columns: Optional[List[str]],
                            screen_monitor: Optional[ScreenMonitor]) -> BaseEyeTrackingParser:
    if et_type.lower() in ["tobii", "tobii csv", "tobii_csv"]:
        from DataParser.TobiiCSVEyeTrackingParser import TobiiCSVEyeTrackingParser
        return TobiiCSVEyeTrackingParser(screen_monitor=screen_monitor, additional_columns=additional_columns)
    raise ValueError(f"Unknown eye tracker type: {et_type}")
