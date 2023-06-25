import numpy as np
import pandas as pd

import Utils.angle_utils as angle_utils
from GazeEvents.SaccadeEvent import SaccadeEvent


class LWSSaccadeEvent(SaccadeEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray,
                 viewer_distance: float, pixel_size: float):
        super().__init__(timestamps=timestamps, x=x, y=y)
        self.__viewer_distance = viewer_distance  # in cm    # TODO: make this a class attribute
        self.__pixel_size = pixel_size                       # TODO: make this a class attribute

    @property
    def visual_angle(self) -> float:
        return angle_utils.calculate_visual_angle(p1=self.start_point, p2=self.end_point, d=self.__viewer_distance,
                                                  pixel_size=self.__pixel_size)

    @property
    def angular_velocity(self) -> float:
        return self.visual_angle / self.duration * 1000

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
        series["visual_angle"] = self.visual_angle
        return series

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not self.__viewer_distance == other.__viewer_distance:
            return False
        if not self.__pixel_size == other.__pixel_size:
            return False
        return True
