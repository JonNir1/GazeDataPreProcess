import pandas as pd
from LWS.DataModels.LWSEnums import LWSStimulusTypeEnum


class LWSBehavioralData:
    """
    Represents the behavioral data for a single trial in the LWS Demo experiment.
    """
    # TODO: decode+encode as pkl file

    def __init__(self, behavioral_data: pd.DataFrame):
        self.__behavioral_data = behavioral_data

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
        return self.__behavioral_data.columns.to_list()

    def get(self, column: str) -> pd.Series:
        return self.__behavioral_data[column]
