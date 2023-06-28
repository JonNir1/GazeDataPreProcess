from abc import ABC
import numpy as np
import pandas as pd

from GazeEvents.BaseGazeEvent import BaseGazeEvent


class BaseVisualGazeEvent(BaseGazeEvent, ABC):
    """
    A base class for Gaze Events that contain visual information, i.e. events that require X, Y coordinates.
    For example, saccades, fixations, smooth pursuit, etc.
    """

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, viewer_distance: float):
        if not np.isfinite(viewer_distance) or viewer_distance <= 0:
            raise ValueError("viewer_distance must be a positive finite number")
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        super().__init__(timestamps=timestamps)
        self._viewer_distance = viewer_distance  # in cm
        self._x = x
        self._y = y
        self.velocities = self.__calculate_velocities()

    @property
    def max_velocity(self) -> float:
        """ Returns the maximum velocity of the event in pixels per second """
        return float(np.nanmax(self.velocities)) * 1000

    @property
    def mean_velocity(self) -> float:
        """ Returns the mean velocity of the event in pixels per second """
        return float(np.nanmean(self.velocities)) * 1000

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - max_velocity: the maximum velocity of the event in pixels per second
            - mean_velocity: the mean velocity of the event in pixels per second
        """
        series = super().to_series()
        series["max_velocity"] = self.max_velocity
        series["mean_velocity"] = self.mean_velocity
        return series

    def __calculate_velocities(self) -> np.ndarray:
        dx = np.diff(self._x)
        dy = np.diff(self._y)
        dt = np.diff(self._timestamps)
        return np.sqrt(dx ** 2 + dy ** 2) / dt

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not super().__eq__(other):
            return False
        if self._viewer_distance != other._viewer_distance:
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        return True

