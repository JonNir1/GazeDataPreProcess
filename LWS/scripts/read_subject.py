# LWS PreProcessing Pipeline

import os
import re
from typing import List

import experiment_config as cnfg
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.scripts.read_behavioral_data import read_behavioral_data


def read_subject_trials(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR, **kwargs) -> List[LWSTrial]:
    """
    Reads the subject's behavioral data and creates a list of trials, each containing the subject's behavioral data,
    the stimulus that was presented in that trial, and the subject's info.

    :param subject_dir: the directory in which the subject's data is stored.
    :param stimuli_dir: the directory in which the stimuli are stored.

    :keyword screen_monitor: a ScreenMonitor object that contains the screen's parameters.
    :keyword experiment_columns or additional_columns: a list of columns that should be read from the behavioral data.
    :keyword start_trigger, end_trigger: the triggers that mark the beginning and end of each trial.

    :return: a list of (unprocessed) LWSTrial objects.

    :raise NotADirectoryError: if the provided subject or stimuli directory does not exist.
    :raise FileNotFoundError: if no subject info files, gaze data files or trigger-log files were found in the provided subject directory.
    :raise ValueError: if multiple subject info files were found in the provided subject directory.
    :raise ValueError: if the number of gaze files and trigger files does not match.
    """
    # verify inputs:
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")

    # read the behavioral data:
    trials = []
    subject_info = read_subject_info(subject_dir)
    behavioral_trials = read_behavioral_data(subject_dir, **kwargs)
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
