import numpy as np
from abc import ABC, abstractmethod
from math import ceil, floor
from typing import List, Tuple

import constants as c
from Config import experiment_config as cnfg


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

    def detect_monocular(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects events in the given gaze data from a single eye
        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye
        :return: array of booleans, where True indicates an event
        :raises ValueError: if x and y are not of equal lengths
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        is_event_candidate = self._find_candidates(x, y)
        event_start_end_idxs = self._find_event_start_end_indices(is_event_candidate)

        # convert to boolean array
        is_event = np.zeros(len(x), dtype=bool)
        if len(event_start_end_idxs) == 0:
            return is_event
        event_idxs = np.concatenate([np.arange(start, end + 1) for start, end in event_start_end_idxs])
        is_event[event_idxs] = True
        return is_event

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
            - 'left': detect events using left eye data only
            - 'right': detect events using right eye data only
            - 'most': detect events using the eye with the most events

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

    @abstractmethod
    def _find_candidates(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array of the same length as the input data, where True indicates the samples that are
        candidates for being part of an event.
        The logic for identifying candidates is specific to each event type and identifier.

        :param x: x coordinates of gaze data
        :param y: y coordinates of gaze data
        """
        raise NotImplementedError

    def _find_event_start_end_indices(self, is_candidate: np.ndarray) -> List[Tuple[int, int]]:
        """
        Every group of consecutive samples that are candidates for being part of an event is considered an event.
        This method returns a list of tuples, where each tuple contains the start and end indices of an event. The
        indices are inclusive.
        Events that are shorter than the minimum duration of an event are excluded from the list.

        :param is_candidate: boolean array indicating whether a sample is a saccade candidate
        :return: list of tuples, each tuple containing the start and end indices of an event
        """
        # if there are no event candidates, return empty list
        if not is_candidate.any():
            return []

        # split candidates to separate events
        candidate_idxs = np.nonzero(is_candidate)[0]
        splitting_idxs = np.where(np.diff(candidate_idxs) > self._min_samples_between_events)[0] + 1  # +1 because we want the index after the split
        separate_event_idxs = np.split(candidate_idxs, splitting_idxs)

        # exclude events that are shorter than the minimum duration
        start_end_idxs = list(map(lambda event_idxs: (event_idxs.min(), event_idxs.max()), separate_event_idxs))
        start_end_idxs = list(filter(lambda sac: sac[1] - sac[0] >= self._min_samples_within_event, start_end_idxs))
        return start_end_idxs
