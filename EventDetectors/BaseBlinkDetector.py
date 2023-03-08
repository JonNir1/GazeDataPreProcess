import numpy as np
from abc import ABC, abstractmethod

import constants as cnst


class BaseBlinkDetector(ABC):
    """
    Baseclass for all blink event detectors.
    Defines three properties:
    - missing_value: default value indicating missing data
    - time_between_blinks: minimum time between two blinks in milliseconds (default: 20)
    - min_duration: minimum duration of a blink in milliseconds (default: 50)
    """

    def __init__(self, missing_value=cnst.MISSING_VALUE,
                 time_between_blinks: float = 20, min_duration: float = 50):
        self.__missing_value = missing_value
        self.__time_between_blinks = time_between_blinks
        self.__min_duration = min_duration

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments

    @property
    def missing_value(self) -> float:
        return self.__missing_value

    @property
    def time_between_blinks(self) -> float:
        return self.__time_between_blinks

    @property
    def min_duration(self) -> float:
        return self.__min_duration


