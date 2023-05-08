import os
import re
import pandas as pd
from typing import List, Tuple

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
        subject_info_paths = self.__find_data_files(subject_dir, "")
        if len(subject_info_paths) == 0:
            raise FileNotFoundError(f"No subject info file was found in {subject_dir}.")
        if len(subject_info_paths) > 1:
            raise ValueError(f"Multiple subject info files were found in {subject_dir}.")
        self.__subject_info = LWSSubjectInfo.from_eprime_file(subject_info_paths[0])

        # extract trial-by-trial data from the gaze and trigger files:
        gaze_paths = self.__find_data_files(subject_dir, "GazeData")
        if len(gaze_paths) == 0:
            raise FileNotFoundError(f"No gaze files were found in {subject_dir}.")
        triggers_paths = self.__find_data_files(subject_dir, "TriggerLog")
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

    @staticmethod
    def __extract_session_from_filename(filename: str) -> int:
        # from a filename like "ExpName-21-33-SomeText.txt" extract the last number (session number; 33 in this case)
        return int(filename.split(".")[0].split("-")[2])

    @staticmethod
    def __find_data_files(subject_dir: str, end_with: str) -> List[str]:
        """
        Find all files in a subject's directory that end with a specific string and match the e-prime file naming
        convention: "<ExpName>-<SubjectID>-<Session>-<DataType>.txt" (e.g. "ExpName-21-33-GazeData.txt")
        """
        pattern = re.compile("[a-zA-z0-9]*-[0-9]+-[0-9]+-" + f"{end_with}.txt")
        paths = [os.path.join(subject_dir, file) for file in os.listdir(subject_dir) if pattern.match(file)]
        paths.sort(key=lambda x: LWSSubject.__extract_session_from_filename(x))
        return paths
