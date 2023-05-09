import pandas as pd

from LWS_DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS_DataModels.LWSArrayStimulus import LWSArrayStimulus


class LWSTrial:
    """
    Represents a single trial in the LWS Demo experiment.
    """

    # TODO: encode as hdf5 file

    def __init__(self,
                 trial_num: int,
                 sr: float,
                 subject_info: LWSSubjectInfo,
                 stimulus: LWSArrayStimulus,
                 behavioral_data: pd.DataFrame):
        self.__trial_num = trial_num
        self.__sampling_rate = sr
        self.__subject_info = subject_info
        self.__stimulus = stimulus
        self.__behavioral_data = behavioral_data

    @property
    def trial_num(self) -> int:
        return self.__trial_num

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def subject_info(self) -> LWSSubjectInfo:
        return self.__subject_info

    @property
    def stimulus(self) -> LWSArrayStimulus:
        return self.__stimulus

    @property
    def behavioral_data(self) -> pd.DataFrame:
        return self.__behavioral_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_S{self.__subject_info.subject_id}_T{self.__trial_num}"

    def __str__(self) -> str:
        return self.__repr__()
