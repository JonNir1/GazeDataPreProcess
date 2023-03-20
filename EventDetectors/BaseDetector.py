from abc import ABC
from math import ceil, floor

import constants as c
import experiment_config as conf


class BaseDetector(ABC):
    """
    Baseclass for all gaze-event detectors.
    Defines these properties:
    - sr: sampling rate of the data in Hz                (default: experiment_config.SAMPLING_RATE)
    - iet: minimal time between two (same) events in ms  (default: experiment_config.INTER_EVENT_TIME)
    """

    def __init__(self, min_duration: float, sr: float = conf.SAMPLING_RATE, iet: float = conf.INTER_EVENT_TIME):
        self.__min_duration = min_duration
        self.__sampling_rate = sr
        self.__inter_event_time = iet

    @property
    def min_duration(self) -> float:
        return self.__min_duration

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def inter_event_time(self) -> float:
        return self.__inter_event_time

    def set_min_duration(self, min_duration: float):
        self.__min_duration = min_duration

    def set_sampling_rate(self, sampling_rate: float):
        self.__sampling_rate = sampling_rate

    def set_inter_event_time(self, inter_event_time: float):
        self.__inter_event_time = inter_event_time

    @property
    def _min_samples_within_event(self) -> int:
        """
        Defines the minimal number of samples required to identify an event.
        """
        return max(1, floor(self.min_duration * self.sampling_rate / c.MILLISECONDS_PER_SECOND))

    @property
    def _min_samples_between_events(self) -> int:
        """
        Defines the minimal number of samples required to identify two adjacent events as separate events.
        """
        return ceil(self.inter_event_time * self.sampling_rate / c.MILLISECONDS_PER_SECOND)
