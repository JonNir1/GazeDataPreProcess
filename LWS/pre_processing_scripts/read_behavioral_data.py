# LWS PreProcessing Pipeline

import os
import re
import numpy as np
import pandas as pd
from typing import List, Union

import constants as cnst
from Config import experiment_config as cnfg
from DataParser.TobiiCSVEyeTrackingParser import TobiiCSVEyeTrackingParser
from DataParser.EPrimeTriggerLogParser import EPrimeTriggerLogParser
from LWS.DataModels.LWSBehavioralData import LWSBehavioralData


def read_behavioral_data(subject_dir: str, **kwargs) -> List[LWSBehavioralData]:
    """
    Reads the eye-tracking data (Tobii+EPrime CSV format) and trigger data (EPrime tsv format) from the specified paths,
    parses them to a predefined format and merges them into a single dataframe for each trial.
    Lastly,

    :param subject_dir: The directory containing the subject's data.

    :keyword screen_monitor: screen monitor object; if None, will be created from the config file
    :keyword additional_columns: additional columns to parse from the eye-tracking data file; if None, will be taken from the config file
    :keyword start_trigger: trigger indicating start of a trial; if None, will be taken from the config file
    :keyword end_trigger: trigger indicating end of a trial; if None, will be taken from the config file

    :return: A list of LWSBehavioralData objects, one for each trial.

    :raise FileNotFoundError: if no gaze files or no trigger files were found in the provided directory.
    :raise ValueError: if the number of gaze files and trigger files does not match.
    """
    gaze_files = __find_files_by_suffix(subject_dir, "GazeData")
    trigger_files = __find_files_by_suffix(subject_dir, "TriggerLog")

    # verify that the number of gaze files and trigger files match:
    if len(gaze_files) == 0:
        raise FileNotFoundError(f"No gaze files were found in {subject_dir}.")
    if len(trigger_files) == 0:
        raise FileNotFoundError(f"No trigger files were found in {subject_dir}.")
    if len(gaze_files) != len(trigger_files):
        raise ValueError(f"Number of gaze files ({len(gaze_files)}) and trigger files ({len(trigger_files)}) "
                         f"does not match.")

    if len(gaze_files) != 1:
        # TODO: support multiple sessions
        raise NotImplementedError("Multiple sessions for a single subject are not supported yet.")
    trials = parse_gaze_and_triggers(et_path=gaze_files[0], trigger_path=trigger_files[0], **kwargs)
    behavioral_data = [LWSBehavioralData(trial_df) for trial_df in trials]
    return behavioral_data


def parse_gaze_and_triggers(et_path, trigger_path,
                            split_trials: bool = True, **kwargs) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Reads the eye-tracking data (Tobii+EPrime CSV format) and trigger data (EPrime tsv format) from the specified paths,
    parses them to a predefined format and merges them into a single dataframe for each trial.

    :param et_path: path to the eye-tracking data file
    :param trigger_path: path to the trigger-log file
    :param split_trials: if True, will split the data into trials according to the start and end triggers

    :keyword additional_columns: additional columns to parse from the eye-tracking data file; if None, will be taken from the config file
    :keyword start_trigger: trigger indicating start of a trial; if None, will be taken from the config file
    :keyword end_trigger: trigger indicating end of a trial; if None, will be taken from the config file
    """
    # get eye tracking parser:
    additional_columns = kwargs.get('additional_columns', None) or cnfg.ADDITIONAL_COLUMNS
    et_parser = TobiiCSVEyeTrackingParser(additional_columns=additional_columns)

    # get trigger parser:
    start_trigger = kwargs.get('start_trigger', cnfg.STIMULUS_ON_TRIGGER)
    end_trigger = kwargs.get('end_trigger', cnfg.STIMULUS_OFF_TRIGGER)
    trigger_parser = EPrimeTriggerLogParser(start_trigger=start_trigger, end_trigger=end_trigger)

    # parse the data and split it into trials:
    et_dfs = et_parser.parse_and_split(et_path)
    trigger_dfs = trigger_parser.parse_and_split(trigger_path)

    joint_dfs = []
    for et_df, trigger_df in zip(et_dfs, trigger_dfs):
        merged_df = pd.merge_asof(et_df, trigger_df, on=cnst.MILLISECONDS, direction='backward')  # merge DFs on time

        # remove samples from before the start trigger or after the end trigger:
        triggers_trial_column = cnst.TRIAL + '_y'  # column name in the merged DF for the trial number based on triggers
        merged_df = merged_df[merged_df[triggers_trial_column].notnull()]
        merged_df = merged_df.drop(columns=[triggers_trial_column])  # no use for two columns specifying the trial num

        # clean the merged DF:
        gaze_trial_column = cnst.TRIAL + '_x'  # column name in the merged DF for the trial number based on ET data
        merged_df = merged_df.rename(columns={gaze_trial_column: cnst.TRIAL})  # rename the trial column
        same_trigger = merged_df[cnst.TRIGGER].diff() == 0
        merged_df.loc[same_trigger, cnst.TRIGGER] = np.nan  # keep only the first instance of a trigger

        # add the new DF to the list:
        joint_dfs.append(merged_df)

    if split_trials:
        return joint_dfs
    return pd.concat(joint_dfs)


def __find_files_by_suffix(directory: str, end_with: str) -> List[str]:
    """
    Find all files in a directory that end with a specific string and match the e-prime naming convention:
        "<ExpName>-<SubjectID>-<Session>-<DataType>.txt"
    examples:
        - Subject Info from E-Prime: "ExpName-21-33.txt"
        - GazeData from Tobii: "ExpName-21-33-GazeData.txt"
        - TriggerLog from E-Prime: "ExpName-21-33-Trigger-Log.txt"
    """
    if end_with:
        end_with = f"-{end_with}.txt"
    else:
        end_with = ".txt"

    # find all filenames that match the pattern:
    pattern = re.compile("[a-zA-z0-9]*-[0-9]*-[0-9]*" + end_with)
    paths = [os.path.join(directory, file) for file in os.listdir(directory) if pattern.match(file)]

    # sort by session number:
    paths.sort(key=lambda p: int(p.split(".")[0].split("-")[2]))
    return paths

