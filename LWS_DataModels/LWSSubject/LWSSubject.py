import os
import re
import pandas as pd
from typing import List, Tuple

from LWS_DataModels.LWSSubject.scripts.identify_data_files import find_data_files_by_suffix
from LWS_DataModels.LWSSubject.LWSSubjectInfo import LWSSubjectInfo
from DataParser.scripts.parse_and_merge import parse_tobii_gaze_and_triggers


class LWSSubject:
    """
    Represents a single subject in the LWS Demo experiment.
    """

    def __init__(self, subject_dir: str):
        if not os.path.isdir(subject_dir):
            raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
        self.__subject_dir = subject_dir

        # extract subject info from the E-Prime file:
        subject_info_paths = find_data_files_by_suffix(subject_dir, "")
        if len(subject_info_paths) == 0:
            raise FileNotFoundError(f"No subject info file was found in {subject_dir}.")
        if len(subject_info_paths) > 1:
            raise ValueError(f"Multiple subject info files were found in {subject_dir}.")
        self.__subject_info = LWSSubjectInfo.from_eprime_file(subject_info_paths[0])

        # extract trial-by-trial data from the gaze and trigger files:
        gaze_paths = find_data_files_by_suffix(subject_dir, "GazeData")
        if len(gaze_paths) == 0:
            raise FileNotFoundError(f"No gaze files were found in {subject_dir}.")
        triggers_paths = find_data_files_by_suffix(subject_dir, "TriggerLog")
        if len(triggers_paths) == 0:
            raise FileNotFoundError(f"No trigger files were found in {subject_dir}.")
        if len(gaze_paths) != len(triggers_paths):
            raise ValueError(f"Number of gaze files ({len(gaze_paths)}) does not match number of trigger files "
                             f"({len(triggers_paths)}).")
        if len(gaze_paths) > 1:
            sr, tobii_trials = self.__handle_multiple_sessions()
        else:
            sr, tobii_trials = parse_tobii_gaze_and_triggers(gaze_paths, triggers_paths)
        self.__sampling_rate = sr
        self.__tobii_trials = tobii_trials

    def __handle_multiple_sessions(self) -> Tuple[float, List[pd.DataFrame]]:
        """
        Merges data from multiple sessions to a single value of sampling rate and a single list of DataFrame trials.
        """
        raise NotImplementedError
