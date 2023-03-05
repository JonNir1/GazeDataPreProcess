import pandas as pd
from abc import ABC, abstractmethod


class BaseGazeDataParser(ABC):

    def __init__(self, path: str):
        self.path = path

    @property
    @abstractmethod
    def SAMPLING_RATE(self) -> float:
        # sampling rate of the data
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MISSING_VALUE(cls) -> float:
        # default value for missing data
        raise NotImplementedError

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
    def RIGHT_X_COLUMN(cls) -> str:
        # column name for right eye x coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        # column name for right eye y coordinate
        raise NotImplementedError

    @abstractmethod
    def parse_gaze_data(self) -> pd.DataFrame:
        raise NotImplementedError



