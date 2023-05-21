import pandas as pd
from typing import Tuple, List, Union, Optional

import constants as cnst
from Utils.calculate_sampling_rate import calculate_sampling_rate_from_microseconds
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
    def sampling_rate(self) -> float:
        microseconds = self.get(cnst.MICROSECONDS)
        return calculate_sampling_rate_from_microseconds(microseconds)

    @property
    def trial_num(self) -> int:
        return self.get(cnst.TRIAL)[0]

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

    def get(self, columns: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        # Returns the requested column(s) from the data
        return self.__data[columns]

    def concat(self, extra_data: Union[pd.DataFrame, pd.Series],
               deep_copy: bool = False) -> Optional["LWSBehavioralData"]:
        """
        Concatenates the extra data to the end of the current data.
        If `deep_copy` is True, returns a new LWSBehavioralData object with the concatenated data. Otherwise, the data
            is concatenated in-place and returns None.
        """
        new_df = pd.concat([self.__data, extra_data], axis=1)
        if deep_copy:
            return LWSBehavioralData(new_df)
        self.__data = new_df

    def __len__(self) -> int:
        return len(self.__data)

    def __eq__(self, other):
        if not isinstance(other, LWSBehavioralData):
            return False
        return self.__data.equals(other.__data)
