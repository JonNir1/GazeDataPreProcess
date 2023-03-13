from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseSaccadeDetector(BaseDetector, ABC):
    """
    Baseclass for all saccade event detectors.
    """

    def __init__(self,
                 min_duration: float = conf.SACCADE_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = conf.INTER_EVENT_TIME):
        super().__init__(sr, iet)
        self.__min_duration = min_duration

    @property
    def min_duration(self) -> float:
        return self.__min_duration

    def set_min_duration(self, min_duration: float):
        self.__min_duration = min_duration
