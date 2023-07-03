# LWS PreProcessing Pipeline

import os

from Config import experiment_config as cnfg
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.pre_processing_scripts.read_raw_data import read_behavioral_data, read_subject_info


def read_subject_from_raw_data(subject_dir: str,
                               stimuli_dir: str = cnfg.STIMULI_DIR,
                               output_directory: str = cnfg.OUTPUT_DIR,
                               **kwargs) -> LWSSubject:
    """
    Reads the subject's behavioral data and creates a list of trials, each containing the subject's behavioral data,
    the stimulus that was presented in that trial, and the subject's info.

    :param subject_dir: the directory in which the subject's data is stored.
    :param stimuli_dir: the directory in which the stimuli are stored.
    :param output_directory: the directory in which all output files will be saved.

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

    # create LWSSubject object:
    subject_info = read_subject_info(subject_dir)
    subject = LWSSubject(info=subject_info, output_directory=output_directory)

    # read the trials and assign them to the subject:
    behavioral_trials_data = read_behavioral_data(subject_dir, **kwargs)
    for i, bd in enumerate(behavioral_trials_data):
        stimulus = LWSArrayStimulus.from_stimulus_name(stim_id=bd.image_num,
                                                       stim_type=bd.stim_type,
                                                       stim_directory=stimuli_dir)
        trial = LWSTrial(trial_num=i + 1, behavioral_data=bd, stimulus=stimulus, subject=subject)
        subject.add_trial(trial)
    return subject
