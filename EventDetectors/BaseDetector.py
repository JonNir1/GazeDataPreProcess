import numpy as np
from abc import ABC, abstractmethod
from math import ceil, floor

import constants as c
import experiment_config as cnfg


class BaseDetector(ABC):
    """
    Baseclass for all gaze-event detectors.
    Defines these properties:
    - sr: sampling rate of the data in Hz
    - min_duration: minimal duration of an event in ms
    - iet: minimal time between two (same) events in ms  (default: 5)
    """

    def __init__(self, sr: float, min_duration: float, iet: float = cnfg.DEFAULT_INTER_EVENT_TIME):
        self.__sampling_rate = sr
        self.__min_duration = min_duration
        self.__inter_event_time = iet

    @abstractmethod
    def detect_monocular(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects events in the given gaze data from a single eye
        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye
        :return: array of booleans, where True indicates an event
        """
        raise NotImplementedError

    def detect_binocular(self,
                         x_l: np.ndarray, y_l: np.ndarray,
                         x_r: np.ndarray, y_r: np.ndarray,
                         detect_by: str = 'both') -> np.ndarray:
        """
        Detects events in the given gaze data from both eyes
        :param x_l, y_l: x- and y-coordinates of gaze data from the left eye
        :param x_r, y_r: x- and y-coordinates of gaze data from the right eye
        :param detect_by: defines how to detect events based on the data from both eyes:
            - 'both'/'and': events are detected if both eyes detect an event
            - 'either'/'or': events are detected if either eye detects an event
            - 'left': events are detected if the left eye detects an event
            - 'right': events are detected if the right eye detects an event
            - 'most': events are detected from the eye with the most samples identified as an event

        :return: array of booleans, where True indicates an event
        """
        is_event_left = self.detect_monocular(x=x_l, y=y_l)
        is_event_right = self.detect_monocular(x=x_r, y=y_r)

        detect_by = detect_by.lower()
        if detect_by in ['both', 'and']:
            return np.logical_and(is_event_left, is_event_right)

        if detect_by in ['either', 'or']:
            return np.logical_or(is_event_left, is_event_right)

        if detect_by == 'left':
            return is_event_left

        if detect_by == 'right':
            return is_event_right

        if detect_by == 'most':
            n_left = np.sum(is_event_left)
            n_right = np.sum(is_event_right)
            if n_left > n_right:
                return is_event_left
            return is_event_right

        raise ValueError("Unknown detect_by value: {}".format(detect_by))

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
