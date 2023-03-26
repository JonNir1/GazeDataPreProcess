import numpy as np
from typing import List

from GazeEvents.BaseEvent import BaseEvent


class FixationEvent(BaseEvent):

    @staticmethod
    def extract_fixation_events(timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, is_fixation: np.ndarray,
                                sampling_rate: float) -> List["FixationEvent"]:
        """
        Extracts fixation events from the given data and returns a list of FixationEvent objects.
        """
        if len(timestamps) != len(x) or len(timestamps) != len(y) or len(timestamps) != len(is_fixation):
            raise ValueError("Arrays of timestamps, x, y and is_fixation must have the same length")
        different_event_idxs = BaseEvent._split_samples_between_events(is_fixation)
        fixation_events = [FixationEvent(timestamps=timestamps[idxs],
                                         sampling_rate=sampling_rate,
                                         x=x[idxs],
                                         y=y[idxs]) for idxs in different_event_idxs]
        return fixation_events

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


