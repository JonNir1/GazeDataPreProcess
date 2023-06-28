import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
from Utils import angle_utils as angle_utils
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent


class SaccadeEvent(BaseVisualGazeEvent):

    @classmethod
    def event_type(cls):
        return cnst.SACCADE

    @property
    def is_outlier(self) -> bool:
        if self.duration < cnfg.DEFAULT_SACCADE_MINIMUM_DURATION:
            return True
        # TODO: check min, max velocity
        return False

    @property
    def start_point(self) -> Tuple[float, float]:
        # returns the saccade's start point as a tuple of the X,Y coordinates
        return self._x[0], self._y[0]

    @property
    def end_point(self) -> Tuple[float, float]:
        # returns the saccade's end point as a tuple of the X,Y coordinates
        return self._x[-1], self._y[-1]

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
    def amplitude(self) -> float:
        return angle_utils.calculate_visual_angle(p1=self.start_point, p2=self.end_point,
                                                  d=self._viewer_distance, pixel_size=cnfg.SCREEN_MONITOR.pixel_size)

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
            - amplitude: saccade's visual angle (in degrees)
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["azimuth"] = self.azimuth
        series["amplitude"] = self.amplitude
        return series
