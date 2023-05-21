import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional

import constants as cnst
from DataParser.BaseParser import BaseParser
from Utils.ScreenMonitor import ScreenMonitor


class BaseEyeTrackingParser(BaseParser, ABC):
    """
    Base class for all parsers handling eye tracking data.
    These parsers take inputs from different eye trackers and map the data to a common format,
    using the method `parse` for parsing the data and `parse_and_split` for splitting the data into trials.
    """

    def __init__(self, additional_columns: Optional[List[str]] = None, screen_monitor: Optional[ScreenMonitor] = None):
        self.__additional_columns: List[str] = additional_columns if additional_columns is not None else []
        self.__screen_monitor: ScreenMonitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()

    def parse(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'File not found: {input_path}')
        df = pd.read_csv(input_path, sep='\t')
        columns_to_keep = self.get_common_columns() + self.__additional_columns
        df.drop(columns=[col for col in df.columns if col not in columns_to_keep], inplace=True)
        df.replace(to_replace=self.MISSING_VALUE(), value=np.nan, inplace=True)

        # correct for screen resolution
        screen_w, screen_h = self.__screen_monitor.resolution
        df[self.LEFT_X_COLUMN()] = df[self.LEFT_X_COLUMN()] * screen_w
        df[self.LEFT_Y_COLUMN()] = df[self.LEFT_Y_COLUMN()] * screen_h
        df[self.RIGHT_X_COLUMN()] = df[self.RIGHT_X_COLUMN()] * screen_w
        df[self.RIGHT_Y_COLUMN()] = df[self.RIGHT_Y_COLUMN()] * screen_h

        # reorder + rename columns to match the standard (except for the additional columns)
        df = df[columns_to_keep]
        df.rename(columns=lambda col: self._column_name_mapper(col), inplace=True)

        if output_path is not None:
            # TODO: implement save_data
            pass
        return df

    def parse_and_split(self, input_path: str, output_path: Optional[str] = None) -> List[pd.DataFrame]:
        df = self.parse(input_path, output_path)
        trial_indices = df[cnst.TRIAL].unique()
        if output_path is not None:
            # TODO: implement save_data
            pass
        return [df[df[cnst.TRIAL] == trial_idx] for trial_idx in trial_indices]

    @classmethod
    def get_common_columns(cls):
        return [cls.TRIAL_COLUMN(), cls.MILLISECONDS_COLUMN(), cls.MICROSECONDS_COLUMN(),
                cls.LEFT_X_COLUMN(), cls.LEFT_Y_COLUMN(), cls.LEFT_PUPIL_COLUMN(),
                cls.RIGHT_X_COLUMN(), cls.RIGHT_Y_COLUMN(), cls.RIGHT_PUPIL_COLUMN()]

    @classmethod
    @abstractmethod
    def MISSING_VALUE(cls) -> float:
        # default value for missing data
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def TRIAL_COLUMN(cls) -> str:
        # column name for trial number
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MILLISECONDS_COLUMN(cls) -> str:
        # column name for time in milliseconds
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MICROSECONDS_COLUMN(cls) -> str:
        # column name for time in microseconds
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

    @classmethod
    def ADDITIONAL_COLUMNS(cls) -> List[str]:
        # column names for additional data
        return []

    @classmethod
    def _column_name_mapper(cls, column_name: str) -> str:
        # maps column names to constants
        if column_name == cls.TRIAL_COLUMN():
            return cnst.TRIAL
        if column_name == cls.MILLISECONDS_COLUMN():
            return cnst.MILLISECONDS
        if column_name == cls.MICROSECONDS_COLUMN():
            return cnst.MICROSECONDS
        if column_name == cls.LEFT_X_COLUMN():
            return cnst.LEFT_X
        if column_name == cls.LEFT_Y_COLUMN():
            return cnst.LEFT_Y
        if column_name == cls.LEFT_PUPIL_COLUMN():
            return cnst.LEFT_PUPIL
        if column_name == cls.RIGHT_X_COLUMN():
            return cnst.RIGHT_X
        if column_name == cls.RIGHT_Y_COLUMN():
            return cnst.RIGHT_Y
        if column_name == cls.RIGHT_PUPIL_COLUMN():
            return cnst.RIGHT_PUPIL
        return column_name
