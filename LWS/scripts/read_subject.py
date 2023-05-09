import os
import re
import pandas as pd
from typing import Tuple, List

import experiment_config as cnfg
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS.DataModels.LWSBehavioralData import LWSBehavioralData
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSTrial import LWSTrial
from DataParser.scripts.parse_and_merge import parse_tobii_gaze_and_triggers


def read_subject(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR) -> Tuple[float, List[LWSTrial]]:
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")

    trials = []
    subject_info = _read_subject_info(subject_dir)
    sr, trial_dataframes = _read_behavioral_data(subject_dir)
    for i, trial_df in enumerate(trial_dataframes):
        behavioral_data = LWSBehavioralData(trial_df)
        stimulus = LWSArrayStimulus.from_stimulus_name(stim_id=behavioral_data.image_num,
                                                       stim_type=behavioral_data.stim_type,
                                                       stim_directory=stimuli_dir)
        lws_trial = LWSTrial(trial_num=i+1, subject_info=subject_info, behavioral_data=behavioral_data, stimulus=stimulus)
        trials.append(lws_trial)
    return sr, trials


def _read_subject_info(subject_dir: str) -> LWSSubjectInfo:
    subject_info_paths = __find_files_by_suffix(subject_dir, "")
    if len(subject_info_paths) == 0:
        raise FileNotFoundError(f"No subject info file was found in {subject_dir}.")
    if len(subject_info_paths) > 1:
        raise ValueError(f"Multiple subject info files were found in {subject_dir}.")
    return LWSSubjectInfo.from_eprime_file(subject_info_paths[0])


def _read_behavioral_data(subject_dir: str) -> Tuple[float, List[pd.DataFrame]]:
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
    sr, tobii_trials = parse_tobii_gaze_and_triggers(gaze_files[0], trigger_files[0])
    return sr, tobii_trials


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
