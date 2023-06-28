import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class FixationEvent(BaseGazeEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, viewer_distance: float):
        if not np.isfinite(viewer_distance) or viewer_distance <= 0:
            raise ValueError("viewer_distance must be a positive finite number")
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        super().__init__(timestamps=timestamps)
        self.__x = x
        self.__y = y
        self.__viewer_distance = viewer_distance  # in cm

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
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
        # TODO: check max acceleration
        # TODO: check max dispersion
        return False

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        # returns the mean coordinates of the fixation on the X,Y axes
        return np.nanmean(self.__x), np.nanmean(self.__y)

    @property
    def std(self) -> Tuple[float, float]:
        # returns the standard deviation of the fixation on the X,Y axes
        return np.nanstd(self.__x), np.nanstd(self.__y)

    @classmethod
    def event_type(cls) -> str:
        return cnst.FIXATION

    def __eq__(self, other):
        if not isinstance(other, FixationEvent):
            return False
        if not super().__eq__(other):
            return False
        if self.__viewer_distance != other.__viewer_distance:
            return False
        if not np.array_equal(self.__x, other.__x, equal_nan=True):
            return False
        if not np.array_equal(self.__y, other.__y, equal_nan=True):
            return False
        return True


