import os
import pandas as pd

import constants as cnst
from Config import experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum

from LWS.pre_processing_scripts.read_subject import read_subject_from_raw_data
from LWS.pre_processing_scripts.visual_angle_to_targets import visual_angle_gaze_to_targets
from LWS.pre_processing_scripts.detect_events import detect_all_events
from LWS.pre_processing_scripts.gen_lws_gaze_events import gen_all_lws_events


def process_subject(subject_dir: str,
                    stimuli_dir: str = cnfg.STIMULI_DIR,
                    output_directory: str = cnfg.OUTPUT_DIR,
                    save_pickle: bool = False, **kwargs) -> LWSSubject:
    """
    For a given subject directory, extracts the subject-info, gaze-data and trigger-log files, and uses those to create
    the LWSTrial objects of that subject. Then, each trial is processed so that we detect blinks, saccades and fixations
    and add the processed data to the trial object.

    :param subject_dir: directory containing the subject's data files.
    :param stimuli_dir: directory containing the stimuli files.
    :param output_directory: directory in which all output files will be saved.
    :param save_pickle: If True, saves the trials' pickle files to the output directory.

    keyword arguments:
        - output_directory: The experiment's output directory, for saving the trials' pickle files if `save_pickle` is True.
        - see gaze detection keyword arguments in `LWS.pre_processing_scripts.detect_events.detect_all_events()`

    :return: A list of LWSTrial objects, one for each trial of the subject, processed and ready to be analyzed.
    """
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")
    if not os.path.isdir(output_directory):
        raise NotADirectoryError(f"Directory {output_directory} does not exist.")

    subject = read_subject_from_raw_data(subject_dir, stimuli_dir, output_directory, **kwargs)
    for i in range(subject.num_trials):
        trial = subject.get_trial(i+1)  # trial numbers start from 1
        process_trial(trial, **kwargs)
    if save_pickle:
        subject.to_pickle()
    return subject


def process_trial(trial: LWSTrial, save_pickle: bool = False, **kwargs):
    """
    Processes the given trial and adds the processed data to the trial object.

    keyword arguments:
        - see gaze detection keyword arguments in `LWS.pre_processing_scripts.detect_events.detect_all_events()`
    """
    trial.is_processed = False
    bd = trial.get_behavioral_data()

    # process raw eye-tracking data
    is_blink, is_saccade, is_fixation = detect_all_events(trial, **kwargs)
    is_event_df = pd.DataFrame({f'is_{GazeEventTypeEnum.BLINK.name.lower()}': is_blink,
                                f'is_{GazeEventTypeEnum.SACCADE.name.lower()}': is_saccade,
                                f'is_{GazeEventTypeEnum.FIXATION.name.lower()}': is_fixation},
                               index=bd.index)

    # calculate visual angles between gaze and targets
    target_distances = visual_angle_gaze_to_targets(trial)
    num_targets, _ = target_distances.shape
    target_distances_df = pd.DataFrame(
        {f'{cnst.DISTANCE}_{cnst.TARGET}{i + 1}': target_distances[i] for i in range(num_targets)},
        index=bd.index)

    # add the new columns to the behavioral data:
    new_behavioral_data = bd.concat(is_event_df, target_distances_df)
    trial.set_behavioral_data(new_behavioral_data)

    # process gaze events
    drop_outlier_events = kwargs.pop('drop_outlier_events', False)
    events = gen_all_lws_events(trial, drop_outliers=drop_outlier_events)
    trial.set_gaze_events(events)

    trial.is_processed = True
