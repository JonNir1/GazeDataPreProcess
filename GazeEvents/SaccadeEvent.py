import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
from Utils import angle_utils as angle_utils
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class SaccadeEvent(BaseGazeEvent):

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
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
            - distance: saccade's distance (in pixels)
            - velocity: saccade's velocity (in pixels per second)
            - azimuth: saccade's azimuth (in degrees)
            - visual_angle: saccade's visual angle (in degrees)
            - angular_velocity: saccade's angular velocity (in degrees per second)
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["azimuth"] = self.azimuth
        series["visual_angle"] = self.visual_angle
        series["angular_velocity"] = self.angular_velocity
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
    def distance(self) -> float:
        # returns the distance of the saccade in pixels
        x_start, y_start = self.start_point
        x_end, y_end = self.end_point
        return np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    @property
    def velocity(self) -> float:
        # returns the velocity of the saccade in pixels per second
        return self.distance / self.duration * 1000

    @property
    def azimuth(self) -> float:
        # returns the azimuth of the saccade in degrees
        # see Utils.angle_utils.calculate_azimuth for more information
        return angle_utils.calculate_azimuth(p1=self.start_point, p2=self.end_point, use_radians=False)

    @property
    def visual_angle(self) -> float:
        return angle_utils.calculate_visual_angle(p1=self.start_point, p2=self.end_point, d=self.__viewer_distance,
                                                  pixel_size=cnfg.SCREEN_MONITOR.pixel_size)

    @property
    def angular_velocity(self) -> float:
        return self.visual_angle / self.duration * 1000

    @classmethod
    def event_type(cls):
        return cnst.SACCADE

    def __eq__(self, other):
        if not isinstance(other, SaccadeEvent):
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
