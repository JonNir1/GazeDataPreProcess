import numpy as np
import pandas as pd

import Utils.angle_utils as angle_utils
from Utils.ScreenMonitor import ScreenMonitor
from GazeEvents.SaccadeEvent import SaccadeEvent


class LWSSaccadeEvent(SaccadeEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray,
                 sampling_rate: float, distance: float, screen_monitor: ScreenMonitor):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate, x=x, y=y)
        self.__distance = distance
        self.__screen_monitor = screen_monitor

    @property
    def visual_angle(self) -> float:
        return angle_utils.calculate_visual_angle(p1=self.start_point, p2=self.end_point,
                                                  d=self.__distance, screen_monitor=self.__screen_monitor)

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - sampling_rate: the sampling rate used to record the event
            - is_outlier: boolean indicating whether the event is an outlier or not
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
            - azimuth: saccade's azimuth in degrees
            - visual_angle: saccade's visual angle in degrees
        """
        series = super().to_series()
        series["visual_angle"] = self.visual_angle
        return series
