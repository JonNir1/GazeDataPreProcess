import numpy as np
import pandas as pd

import constants as cnst
import experiment_config as cnfg
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class FixationEvent(BaseGazeEvent):

    def __init__(self, timestamps: np.ndarray, sampling_rate: float, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate)
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        self.__x = x
        self.__y = y

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the following index:
            - start_time: fixation's start time in milliseconds
            - end_time: fixation's end time in milliseconds
            - duration: fixation's duration in milliseconds
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - std: fixation's standard deviation (in pixels units)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["std"] = self.std
        return series

    @property
    def is_outlier(self) -> bool:
        if self.duration < cnfg.DEFAULT_FIXATION_MINIMUM_DURATION:
            return True
        # TODO: check max velocity
        return False

    @property
    def center_of_mass(self) -> np.ndarray:
        # returns a 2D array of the mean X,Y coordinates of the fixation
        return np.array([np.nanmean(self.__x), np.nanmean(self.__y)])

    @property
    def std(self) -> np.ndarray:
        # returns a 2D array of the standard deviation of the X,Y coordinates of the fixation
        return np.array([np.nanstd(self.__x), np.nanstd(self.__y)])

    @classmethod
    def event_type(cls) -> str:
        return cnst.FIXATION

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__x, other.__x):
            return False
        if not np.array_equal(self.__y, other.__y):
            return False
        return True


