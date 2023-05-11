import pandas as pd
from typing import Tuple
from LWS.DataModels.LWSEnums import LWSStimulusTypeEnum


class LWSBehavioralData:
    """
    Represents the behavioral data for a single trial in the LWS Demo experiment.
    """
    # TODO: decode+encode as pkl file

    def __init__(self, data: pd.DataFrame):
        self.__data = data

    @property
    def shape(self) -> Tuple[int, int]:
        return self.__data.shape

    @property
    def trial_num(self) -> int:
        return self.get("trial")[0]

    @property
    def stim_type(self) -> LWSStimulusTypeEnum:
        stim_type_str = self.get("ConditionName")[0]
        return LWSStimulusTypeEnum(stim_type_str)

    @property
    def image_num(self) -> int:
        return self.get("ImageNum")[0]

    @property
    def columns(self) -> list:
        return self.__data.columns.to_list()

    def get(self, column: str) -> pd.Series:
        return self.__data[column]

    def __len__(self) -> int:
        return len(self.__data)

    def __eq__(self, other):
        if not isinstance(other, LWSBehavioralData):
            return False
        return self.__data.equals(other.__data)
