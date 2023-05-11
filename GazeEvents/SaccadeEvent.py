import numpy as np
import pandas as pd

from Utils import velocity_utils as vu
from GazeEvents.BaseEvent import BaseEvent
from Utils.ScreenMonitor import ScreenMonitor


class SaccadeEvent(BaseEvent):

    def __init__(self, timestamps: np.ndarray, sampling_rate: float, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate)
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        self.__x = x
        self.__y = y

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the following index:
            - start_time: saccade's start time in milliseconds
            - end_time: saccade's end time in milliseconds
            - duration: saccade's duration in milliseconds
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        return series

    @property
    def start_point(self) -> np.ndarray:
        # returns a 2D array of the X,Y coordinates of the saccade's start point
        return np.array([self.__x[0], self.__y[0]])

    @property
    def end_point(self) -> np.ndarray:
        # returns a 2D array of the X,Y coordinates of the saccade's end point
        return np.array([self.__x[-1], self.__y[-1]])

    def calculate_mean_velocity(self, d: float, screen_monitor: ScreenMonitor) -> float:
        """
        Calculates the mean velocity of the saccade in degrees per second.
        :param d: the distance between the screen and the participant's eyes in centimeters
        :param screen_monitor: the ScreenMonitor object that holds information about the screen used in the experiment
        """
        # returns the mean velocity of the saccade in degrees per second
        x_s, y_s = self.start_point
        x_e, y_e = self.end_point
        angle = screen_monitor.calc_angle_between_pixels(d=d, p1=(x_s, y_s), p2=(x_e, y_e), use_radians=False)
        return 1000 * angle / self.duration

    def calculate_peak_velocity(self, d: float, screen_monitor: ScreenMonitor) -> float:
        """
        Calculates the maximum velocity of the saccade in degrees per second.
        :param d: the distance between the screen and the participant's eyes in centimeters
        :param screen_monitor: the ScreenMonitor object that holds information about the screen used in the experiment
        """
        velocities = vu.calculate_angular_velocity(x=self.__x, y=self.__y,
                                                   sr=self.__sampling_rate, d=d,
                                                   screen_monitor=screen_monitor,
                                                   use_radians=False)
        return np.nanmax(velocities)

    @classmethod
    def _event_type(cls):
        return "saccade"

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__x, other.__x):
            return False
        if not np.array_equal(self.__y, other.__y):
            return False
        return True
