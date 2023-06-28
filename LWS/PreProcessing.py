import os
import pandas as pd
from typing import List

import constants as cnst
from Config import experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial

from LWS.pre_processing_scripts.read_subject import read_subject_trials
from LWS.pre_processing_scripts.visual_angle_to_targets import calculate_visual_angle_between_gaze_data_and_targets
from LWS.pre_processing_scripts.detect_events import detect_all_events
from LWS.pre_processing_scripts.gen_lws_gaze_events import gen_all_lws_events


def process_subject(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR,
                    save_pickle: bool = False, **kwargs) -> List[LWSTrial]:
    """
    For a given subject directory, extracts the subject-info, gaze-data and trigger-log files, and uses those to create
    the LWSTrial objects of that subject. Then, each trial is processed so that we detect blinks, saccades and fixations
    and add the processed data to the trial object.

    :param subject_dir: directory containing the subject's data files.
    :param stimuli_dir: directory containing the stimuli files.
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

    trials = read_subject_trials(subject_dir, stimuli_dir, **kwargs)
    for _i, trial in enumerate(trials):
        process_trial(trial, save_pickle, **kwargs)
    return trials


def process_trial(trial: LWSTrial, save_pickle: bool = False, **kwargs):
    """
    Processes the given trial and adds the processed data to the trial object.

    keyword arguments:
        - output_directory: The experiment's output directory, for saving the trial's pickle file if `save_pickle` is True.
        - see gaze detection keyword arguments in `LWS.pre_processing_scripts.detect_events.detect_all_events()`
    """
    trial.is_processed = False
    bd = trial.get_behavioral_data()

    # process raw eye-tracking data
    is_blink, is_saccade, is_fixation = detect_all_events(trial, **kwargs)
    target_distance = calculate_visual_angle_between_gaze_data_and_targets(trial)
    is_event_df = pd.DataFrame({f'is_{cnst.BLINK}': is_blink, f'is_{cnst.SACCADE}': is_saccade,
                                f'is_{cnst.FIXATION}': is_fixation, f'{cnst.TARGET}_{cnst.DISTANCE}': target_distance},
                               index=bd.index)

    # add the new columns to the behavioral data:
    new_behavioral_data = bd.concat(is_event_df)
    trial.set_behavioral_data(new_behavioral_data)

    # process gaze events
    drop_outlier_events = kwargs.pop('drop_outlier_events', False)
    events = gen_all_lws_events(trial, sm, drop_outliers=drop_outlier_events)
    trial.set_gaze_events(events)

    trial.is_processed = True

    if save_pickle:
        output_dir = kwargs.pop('output_directory', cnfg.OUTPUT_DIR)
        trial.to_pickle(output_dir)
