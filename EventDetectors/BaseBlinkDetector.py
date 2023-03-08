import numpy as np
from abc import ABC, abstractmethod

import experiment_config as conf


class BaseBlinkDetector(ABC):
    """
    Baseclass for all blink event detectors.
    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz                        (default: experiment_config.SAMPLING_RATE)
    - missing_value: default value indicating missing data                  (default: experiment_config.MISSING_VALUE)
    - time_between_blinks: minimum time between two blinks in milliseconds  (default: 20)
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    """

    def __init__(self, time_between_blinks: float = 20, min_duration: float = 50):
        self.__sampling_rate = conf.SAMPLING_RATE
        self.__missing_value = conf.MISSING_VALUE
        self.__time_between_blinks = time_between_blinks
        self.__min_duration = min_duration

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    def set_sampling_rate(self, sampling_rate: float):
        self.__sampling_rate = sampling_rate

    @property
    def missing_value(self) -> float:
        return self.__missing_value

    def set_missing_value(self, missing_value: float):
        self.__missing_value = missing_value

    @property
    def time_between_blinks(self) -> float:
        return self.__time_between_blinks

    @property
    def min_duration(self) -> float:
        return self.__min_duration


