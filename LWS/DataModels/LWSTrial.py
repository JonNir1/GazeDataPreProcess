import pandas as pd

from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSBehavioralData import LWSBehavioralData


class LWSTrial:
    """
    Represents a single trial in the LWS Demo experiment.
    """

    # TODO: encode as hdf5 file

    def __init__(self,
                 trial_num: int,
                 subject_info: LWSSubjectInfo,
                 stimulus: LWSArrayStimulus,
                 behavioral_data: LWSBehavioralData):
        self.__is_processed = False
        self.__trial_num = trial_num
        self.__subject_info = subject_info
        self.__stimulus = stimulus
        self.__behavioral_data = behavioral_data

    @property
    def trial_num(self) -> int:
        return self.__trial_num

    @property
    def subject_info(self) -> LWSSubjectInfo:
        return self.__subject_info

    @property
    def stimulus(self) -> LWSArrayStimulus:
        return self.__stimulus

    @property
    def is_processed(self) -> bool:
        return self.__is_processed

    def set_is_processed(self, is_processed: bool):
        if self.__is_processed and not is_processed:
            raise RuntimeError("Cannot set is_processed to False after it has been set to True.")
        self.__is_processed = is_processed

    @property
    def behavioral_data(self) -> LWSBehavioralData:
        return self.__behavioral_data

    def set_behavioral_data(self, behavioral_data: LWSBehavioralData):
        if self.is_processed:
            raise RuntimeError("Cannot set behavioral data after trial has been processed.")
        self.__behavioral_data = behavioral_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_S{self.__subject_info.subject_id}_T{self.__trial_num}"

    def __str__(self) -> str:
        return self.__repr__()
