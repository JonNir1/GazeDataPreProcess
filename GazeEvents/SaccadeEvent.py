import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
import experiment_config as cnfg
from Utils import angle_utils as angle_utils
from Utils.ScreenMonitor import ScreenMonitor
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class SaccadeEvent(BaseGazeEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps=timestamps)
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        self.__x = x
        self.__y = y

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
            - azimuth: saccade's azimuth in degrees
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["azimuth"] = self.azimuth
        return series

    @property
    def is_outlier(self) -> bool:
        if self.duration < cnfg.DEFAULT_SACCADE_MINIMUM_DURATION:
            return True
        # TODO: check min, max velocity
        return False

    @property
    def start_point(self) -> Tuple[float, float]:
        # returns the saccade's start point as a tuple of the X,Y coordinates
        return self.__x[0], self.__y[0]

    @property
    def end_point(self) -> Tuple[float, float]:
        # returns the saccade's end point as a tuple of the X,Y coordinates
        return self.__x[-1], self.__y[-1]

    @property
    def azimuth(self) -> float:
        # returns the azimuth of the saccade in degrees
        # see Utils.angle_utils.calculate_azimuth for more information
        return angle_utils.calculate_azimuth(p1=self.start_point, p2=self.end_point, use_radians=False)

    @classmethod
    def event_type(cls):
        return cnst.SACCADE

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__x, other.__x):
            return False
        if not np.array_equal(self.__y, other.__y):
            return False
        return True
