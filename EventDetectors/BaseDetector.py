from abc import ABC, abstractmethod

import experiment_config as conf


class BaseDetector(ABC):
    """
    Baseclass for all gaze-event detectors.
    Defines these properties:
    - sr: sampling rate of the data in Hz                (default: experiment_config.SAMPLING_RATE)
    - iet: minimal time between two (same) events in ms  (default: experiment_config.INTER_EVENT_TIME)
    """

    def __init__(self, sr: float = conf.SAMPLING_RATE, iet: float = conf.INTER_EVENT_TIME):
        self.__sampling_rate = sr
        self.__inter_event_time = iet

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    def set_sampling_rate(self, sampling_rate: float):
        self.__sampling_rate = sampling_rate

    @property
    def inter_event_time(self) -> float:
        return self.__inter_event_time

    def set_inter_event_time(self, inter_event_time: float):
        self.__inter_event_time = inter_event_time

    @property
    @abstractmethod
    def min_duration(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_min_duration(self, min_duration: float):
        raise NotImplementedError
