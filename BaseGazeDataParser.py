import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List

import constants as cnst


class BaseGazeDataParser(ABC):

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        self.path = path
        num_samples, sampling_rate = self._compute_sample_size_and_sr()
        self.__num_samples = num_samples
        self.__sampling_rate = sampling_rate

    @property
    def num_samples(self) -> float:
        # number of samples in the data
        return self.__num_samples

    @property
    def sampling_rate(self) -> float:
        # sampling rate of the data
        return self.__sampling_rate

    @classmethod
    @abstractmethod
    def MISSING_VALUE(cls) -> float:
        # default value for missing data
        raise NotImplementedError

    @classmethod
    def get_columns(cls) -> List[str]:
        return [cls.TIME_COLUMN(), cls.LEFT_X_COLUMN(), cls.LEFT_Y_COLUMN(), cls.LEFT_PUPIL_COLUMN(),
                cls.RIGHT_X_COLUMN(), cls.RIGHT_Y_COLUMN(), cls.RIGHT_PUPIL_COLUMN()]

    @classmethod
    @abstractmethod
    def TIME_COLUMN(cls) -> str:
        # column name for time
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_X_COLUMN(cls) -> str:
        # column name for left eye x coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_Y_COLUMN(cls) -> str:
        # column name for left eye y coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_PUPIL_COLUMN(cls) -> str:
        # column name for left eye pupil diameter
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_X_COLUMN(cls) -> str:
        # column name for right eye x coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        # column name for right eye y coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_PUPIL_COLUMN(cls) -> str:
        # column name for right eye pupil diameter
        raise NotImplementedError

    @abstractmethod
    def parse_gaze_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _compute_sample_size_and_sr(self) -> (int, float):
        raise NotImplementedError

    def _column_name_mapper(self, column_name):
        if column_name == self.TIME_COLUMN():
            return cnst.TIME
        if column_name == self.LEFT_X_COLUMN():
            return cnst.LEFT_X
        if column_name == self.LEFT_Y_COLUMN():
            return cnst.LEFT_Y
        if column_name == self.LEFT_PUPIL_COLUMN():
            return cnst.LEFT_PUPIL
        if column_name == self.RIGHT_X_COLUMN():
            return cnst.RIGHT_X
        if column_name == self.RIGHT_Y_COLUMN():
            return cnst.RIGHT_Y
        if column_name == self.RIGHT_PUPIL_COLUMN():
            return cnst.RIGHT_PUPIL
        return column_name
