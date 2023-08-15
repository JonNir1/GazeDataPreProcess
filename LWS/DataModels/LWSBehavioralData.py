import numpy as np
import pandas as pd
from typing import Tuple, List, Union

import constants as cnst
from Utils.calculate_sampling_rate import calculate_sampling_rate_from_microseconds


class LWSBehavioralData:
    """
    Represents the behavioral data for a single trial in the LWS Demo experiment.
    """

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
    def columns(self) -> list:
        return self.__data.columns.to_list()

    @property
    def index(self) -> list:
        return self.__data.index.to_list()

    def get(self, columns: Union[str, List[str]]) -> np.ndarray:
        # Returns the requested column(s) from the data
        return self.__data[columns].values

    def concat(self, *extra_data: Union[pd.DataFrame, pd.Series]) -> "LWSBehavioralData":
        """
        Concatenates the extra data to the end of the current data, and returns a new LWSBehavioralData object.
        """
        new_df = pd.concat([self.__data, *extra_data], axis=1)
        return LWSBehavioralData(new_df)

    def __len__(self) -> int:
        return len(self.__data)

    def __eq__(self, other):
        if not isinstance(other, LWSBehavioralData):
            return False
        return self.__data.equals(other.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{str(self.shape)}"

    def __str__(self) -> str:
        trial_num = self.get(cnst.TRIAL)[0]
        return f"{self.__class__.__name__}_T{trial_num:03d}"
