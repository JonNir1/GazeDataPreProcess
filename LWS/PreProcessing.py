import os
import time
import pandas as pd

import constants as cnst
from Config import experiment_config as cnfg
import Utils.io_utils as ioutils
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum

from LWS.PreProcessingScripts.read_subject import read_subject_from_raw_data
from LWS.PreProcessingScripts.visual_angle_to_targets import visual_angle_gaze_to_targets
from LWS.PreProcessingScripts.detect_events import detect_all_events
from LWS.PreProcessingScripts.gen_lws_gaze_events import gen_all_lws_events
from LWS.PreProcessingScripts.create_subject_dataframes import create_subject_dataframes


def process_subject(subject_dir: str,
                    stimuli_dir: str = cnfg.STIMULI_DIR,
                    output_dir: str = cnfg.OUTPUT_DIR,
                    **kwargs) -> LWSSubject:
    """
    For a given subject directory, extracts the subject-info, gaze-data and trigger-log files, and uses those to create
    the LWSTrial objects of that subject. Then, each trial is processed so that we detect blinks, saccades and fixations
    and add the processed data to the trial object.

    :param subject_dir: directory containing the subject's data files.
    :param stimuli_dir: directory containing the stimuli files.
    :param output_dir: directory in which all current/future output files will be saved

    keyword arguments:
        - save_results: If true, saves the processed pickle files to the output directory.
        - perform_subject_analysis: If true, performs the subject post-processing analysis.
        - see gaze detection keyword arguments in `LWS.PreProcessingScripts.detect_events.detect_all_events()`

    :return: A list of LWSTrial objects, one for each trial of the subject, processed and ready to be analyzed.
    """
    start = time.time()
    verbose = kwargs.get('verbose', True)
    if verbose:
        ioutils.print_and_log(msg="###################\n" +
                                  f"Pre-processing subject `{os.path.basename(subject_dir)}`...",
                              log_file=None)

    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Directory {output_dir} does not exist.")

    subject = read_subject_from_raw_data(subject_dir, stimuli_dir, output_dir, **kwargs)
    for tr in subject.get_trials():
        process_trial(tr, **kwargs)

    save_results = kwargs.get('save_results', False)
    if kwargs.get('perform_subject_analysis', False):
        subject_dfs = create_subject_dataframes(subject, save=save_results)

    if save_results:
        subject.to_pickle()

    end = time.time()
    if verbose:
        ioutils.print_and_log(msg=f"Finished preprocessing subject `{str(subject)}`: {(end - start):.2f} seconds",
                              log_file=None)
    return subject


def process_trial(trial: LWSTrial, save_pickle: bool = False, **kwargs):
    """
    Processes the given trial and adds the processed data to the trial object.

    keyword arguments: see gaze detection keyword arguments in
        `LWS.PreProcessingScripts.detect_events.detect_all_events()`
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
    colname_prefix = f"{cnst.DISTANCE}_{cnst.TARGET}"
    target_distances_df = pd.DataFrame(
        {f'{colname_prefix}{i}': target_distances[i] for i in range(num_targets)},
        index=bd.index)
    closest_target = target_distances_df.idxmin(axis=1, skipna=True, numeric_only=True)
    closest_target = closest_target.apply(lambda x: str(x).removeprefix(f"{cnst.DISTANCE}_{cnst.TARGET}")).astype(float)
    target_distances_df["closest_target"] = closest_target

    # add the new columns to the behavioral data:
    new_behavioral_data = bd.concat(is_event_df, target_distances_df)
    trial.set_behavioral_data(new_behavioral_data)

    # process gaze events
    drop_outlier_events = kwargs.pop('drop_outlier_events', False)
    events = gen_all_lws_events(trial, drop_outliers=drop_outlier_events)
    trial.set_gaze_events(events)

    trial.is_processed = True
