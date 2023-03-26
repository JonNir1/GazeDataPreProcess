import numpy as np
from typing import List

from GazeEvents.BaseEvent import BaseEvent


class FixationEvent(BaseEvent):

    def __init__(self, timestamps: np.ndarray, sampling_rate: float, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate)
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        self.__x = x
        self.__y = y

    @property
    def center_of_mass(self) -> np.ndarray:
        # returns a 2D array of the mean X,Y coordinates of the fixation
        return np.array([np.nanmean(self.__x), np.nanmean(self.__y)])

    @property
    def std(self) -> np.ndarray:
        # returns a 2D array of the standard deviation of the X,Y coordinates of the fixation
        return np.array([np.nanstd(self.__x), np.nanstd(self.__y)])

    @classmethod
    def _event_type(cls) -> str:
        return "fixation"

    def __eq__(self, other):
        if not isinstance(other, FixationEvent):
            return False
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__x, other.__x):
            return False
        if not np.array_equal(self.__y, other.__y):
            return False
        return True


