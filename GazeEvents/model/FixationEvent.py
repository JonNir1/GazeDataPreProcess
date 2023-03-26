import numpy as np

from GazeEvents.model.BaseEvent import BaseEvent


class FixationEvent(BaseEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps)
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


