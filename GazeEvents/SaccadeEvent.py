import numpy as np
import pandas as pd

import visual_angle_utils as vau
import velocity_utils as vu
from GazeEvents.BaseEvent import BaseEvent


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
            - mean_velocity: event's mean velocity in degrees per second
            - peak_velocity: event's peak velocity in degrees per second
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["mean_velocity"] = self.mean_velocity
        series["peak_velocity"] = self.peak_velocity
        return series


    @property
    def start_point(self) -> np.ndarray:
        # returns a 2D array of the X,Y coordinates of the saccade's start point
        return np.array([self.__x[0], self.__y[0]])

    @property
    def end_point(self) -> np.ndarray:
        # returns a 2D array of the X,Y coordinates of the saccade's end point
        return np.array([self.__x[-1], self.__y[-1]])

    @property
    def mean_velocity(self) -> float:
        # returns the mean velocity of the saccade in degrees per second
        edge_points = np.array([self.start_point, self.end_point]).reshape((2, 2))
        total_degrees = vau.pixels2deg(edge_points)
        return 1000 * total_degrees / self.duration

    @property
    def peak_velocity(self) -> float:
        # returns the peak velocity between two adjacent samples of the saccade, in degrees per second
        velocities = vu.calculate_angular_velocity(self.__x, self.__y, self.__sampling_rate)
        return np.nanmax(velocities)

    @classmethod
    def _event_type(cls):
        return "saccade"

    def __eq__(self, other):
        if not isinstance(other, SaccadeEvent):
            return False
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__x, other.__x):
            return False
        if not np.array_equal(self.__y, other.__y):
            return False
        return True
