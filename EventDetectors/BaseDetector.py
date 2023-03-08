from abc import ABC

import experiment_config as conf


class BaseDetector(ABC):
    """
    Baseclass for all gaze-event detectors.
    """

    def __init__(self):
        self.__sampling_rate = conf.SAMPLING_RATE
        self.__missing_value = conf.MISSING_VALUE

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
