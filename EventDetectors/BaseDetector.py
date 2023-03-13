from abc import ABC, abstractmethod

import experiment_config as conf


class BaseDetector(ABC):
    """
    Baseclass for all gaze-event detectors.
    Defines these properties:
    - sr: sampling rate of the data in Hz                (default: experiment_config.SAMPLING_RATE)
    """

    def __init__(self, sr: float = conf.SAMPLING_RATE):
        self.__sampling_rate = sr

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    def set_sampling_rate(self, sampling_rate: float):
        self.__sampling_rate = sampling_rate

    @property
    @abstractmethod
    def min_duration(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_min_duration(self, min_duration: float):
        raise NotImplementedError
