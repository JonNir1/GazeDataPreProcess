import numpy as np
from abc import ABC, abstractmethod
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

    INTER_EVENT_TIME = 5  # minimal time between two (same) events in milliseconds (two saccades, two fixations, etc.)

    def __init__(self, sr: float, min_duration: float, iet: float = INTER_EVENT_TIME):
        self.__sampling_rate = sr
        self.__min_duration = min_duration
        self.__inter_event_time = iet

    @abstractmethod
    def detect(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects events in the given gaze data.
        :param x: x-coordinates of gaze data
        :param y: y-coordinates of gaze data
        :return: array of booleans, where True indicates an event
        """
        raise NotImplementedError

    @classmethod
    def event_type(cls):
        class_name = cls.__name__.lower()
        if "blink" in class_name:
            return "blink"
        if "saccade" in class_name:
            return "saccade"
        if "fixation" in class_name:
            return "fixation"
        raise ValueError("Unknown event type for detector: {}".format(class_name))

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
