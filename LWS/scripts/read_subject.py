import os
import re
import pandas as pd
from typing import Optional, List

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.scripts.read_behavioral_data import read_behavioral_data


def read_subject_trials(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR, **kwargs) -> List[LWSTrial]:
    """

    :param subject_dir:
    :param stimuli_dir:

    :keyword screen_monitor:
    :keyword experiment_columns or additional_columns:
    :keyword start_trigger, end_trigger:

    :return:
    """
    # verify inputs:
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")

    # extract keyword arguments:
    sm = kwargs.get("screen_monitor", None) or ScreenMonitor.from_config()
    experiment_columns = kwargs.get("experiment_columns", None) or kwargs.get("additional_columns",
                                                                              None) or cnfg.ADDITIONAL_COLUMNS
    start_trigger = kwargs.get("start_trigger", None) or cnfg.START_TRIGGER
    end_trigger = kwargs.get("end_trigger", None) or cnfg.END_TRIGGER

    trials = []
    subject_info = read_subject_info(subject_dir)
    behavioral_trials = read_behavioral_data(subject_dir, screen_monitor=sm)
    for i, bt in enumerate(behavioral_trials):
        stimulus = LWSArrayStimulus.from_stimulus_name(stim_id=bt.image_num,
                                                       stim_type=bt.stim_type,
                                                       stim_directory=stimuli_dir)
        lws_trial = LWSTrial(trial_num=i + 1, subject_info=subject_info, behavioral_data=bt, stimulus=stimulus)
        trials.append(lws_trial)
    return trials


def read_subject_info(subject_dir: str) -> LWSSubjectInfo:
    """
    Finds all files in the provided directory that match the subject-info pattern ("ExpName-21-33.txt"), reads the only
    file (or raises an error if there are multiple files or no files at all), and extracts a SubjectInfo object from it.

    :raise FileNotFoundError: if no subject info file was found in the provided directory.
    :raise ValueError: if multiple subject info files were found in the provided directory.
    """
    # find all filenames that match the subject-info pattern:
    pattern = re.compile("[a-zA-z0-9]*-[0-9]*-[0-9]*.txt")
    subject_info_paths = [os.path.join(subject_dir, file) for file in os.listdir(subject_dir) if pattern.match(file)]
    subject_info_paths.sort(key=lambda p: int(p.split(".")[0].split("-")[2]))  # sort by session number

    if len(subject_info_paths) == 0:
        raise FileNotFoundError(f"No subject info file was found in {subject_dir}.")
    if len(subject_info_paths) > 1:
        raise ValueError(f"Multiple subject info files were found in {subject_dir}.")
    return LWSSubjectInfo.from_eprime_file(subject_info_paths[0])
