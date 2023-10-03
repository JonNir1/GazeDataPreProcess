import os
import pickle as pkl
from typing import List, Optional

import pandas as pd

from Config import experiment_config as cnfg
import Utils.io_utils as ioutils
from LWS.DataModels.LWSArrayStimulus import LWSStimulusTypeEnum
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo


class LWSSubject:
    """
    Represents a single LWS subject, with it's personal info and experimental data
    """

    def __init__(self, info: LWSSubjectInfo, trials: List["LWSTrial"] = None, output_directory: str = cnfg.OUTPUT_DIR):
        self.__subject_info: LWSSubjectInfo = info
        self.__trials: List[LWSTrial] = trials if trials is not None else []
        self.__output_directory: str = ioutils.create_subject_output_directory(subject_id=self.subject_id,
                                                                               output_dir=output_directory)

    @staticmethod
    def from_pickle(pickle_path: str) -> "LWSSubject":
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Could not find pickle file: {pickle_path}")
        with open(pickle_path, "rb") as f:
            subject = pkl.load(f)
        if not isinstance(subject, LWSSubject):
            raise RuntimeError(f"Expected LWSSubject, got {type(subject)}")
        return subject

    @property
    def subject_id(self) -> int:
        return self.__subject_info.subject_id

    @property
    def output_dir(self) -> str:
        return self.__output_directory

    @property
    def log_file(self) -> str:
        return os.path.join(self.output_dir, f"log.{ioutils.TEXT_EXTENSION}")

    @property
    def dominant_eye(self) -> str:
        return self.__subject_info.dominant_eye

    @property
    def distance_to_screen(self) -> float:
        return self.__subject_info.distance_to_screen

    @property
    def is_processed(self) -> bool:
        return all([trial.is_processed for trial in self.__trials])

    @property
    def num_trials(self) -> int:
        return len(self.__trials)

    def add_trial(self, trial: "LWSTrial"):
        self.__trials.append(trial)

    def get_trials(self, stim_type: Optional[LWSStimulusTypeEnum] = None) -> List["LWSTrial"]:
        """ Returns a list of the subject's trials, optionally filtered by stimulus type """
        all_trials = self.__trials
        if stim_type is None:
            return all_trials
        return list(filter(lambda t: t.stim_type == stim_type, all_trials))

    def to_pickle(self) -> str:
        filename = ioutils.get_filename(name=self.__repr__(), extension=ioutils.PICKLE_EXTENSION)
        full_path = os.path.join(self.output_dir, filename)
        with open(full_path, "wb") as f:
            pkl.dump(self, f)
        return full_path

    def get_dataframe_path(self, df_name: str) -> str:
        return os.path.join(self.output_dir, "dataframes", f"{df_name}.{ioutils.PICKLE_EXTENSION}")

    def _get_full_raw_data(self) -> pd.DataFrame:
        """ Returns a DataFrame with all the raw data from all trials """
        # access the private __data attribute of each trial, and concatenate them all together
        return pd.concat([tr.get_behavioral_data()._LWSBehavioralData__data for tr in self.get_trials()], axis=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.subject_id:03d}"

    def __str__(self) -> str:
        return f"S{self.subject_id:03d}"

    def __eq__(self, other: "LWSSubject") -> bool:
        if not isinstance(other, LWSSubject):
            return False
        if self.subject_id != other.subject_id:
            return False
        if self.is_processed != other.is_processed:
            return False
        if self.__subject_info != other.__subject_info:
            return False
        if self.num_trials != other.num_trials:
            return False
        for i in range(self.num_trials):
            if self.__trials[i] != other.__trials[i]:
                return False
        return True
    
    def __hash__(self):
        return hash(self.__repr__())


# import at the bottom to avoid circular imports
from LWS.DataModels.LWSTrial import LWSTrial



