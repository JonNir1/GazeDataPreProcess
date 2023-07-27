from abc import ABC
import numpy as np
import pandas as pd
from typing import final

import Utils.array_utils as au
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
        self._velocities = self.__calculate_velocities()

    @final
    @property
    def max_velocity(self) -> float:
        """ Returns the maximum velocity of the event in pixels per second """
        return float(np.nanmax(self._velocities)) * 1000

    @final
    @property
    def mean_velocity(self) -> float:
        """ Returns the mean velocity of the event in pixels per second """
        return float(np.nanmean(self._velocities)) * 1000

    @final
    def get_timestamps(self, round_decimals: int = 1, zero_corrected: bool = True) -> np.ndarray:
        """
        Returns the timestamps of the event, rounded to the specified number of decimals.
        If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self._timestamps  # timestamps in milliseconds
        if zero_corrected:
            timestamps = timestamps - timestamps[0]  # start from 0
        timestamps = np.round(timestamps, decimals=round_decimals)
        return timestamps

    @final
    def get_velocity_series(self, round_decimals: int = 1, zero_corrected: bool = True) -> pd.Series:
        """
        Returns a pandas Series with the event's velocities (px/s) and indexed by timestamps, rounded to the specified
        number of decimals. If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self.get_timestamps(round_decimals=round_decimals, zero_corrected=zero_corrected)
        return pd.Series(data=self._velocities, index=timestamps, name="velocity")

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
        distances = au.distance_between_subsequent_pixels(self._x, self._y)
        dt = np.diff(self._timestamps)
        velocities = distances / dt
        return np.concatenate(([np.nan], velocities))  # first velocity is always NaN

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

