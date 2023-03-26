import numpy as np
from abc import ABC, abstractmethod
from typing import List


class BaseEvent(ABC):

    def __init__(self, timestamps: np.ndarray, sampling_rate: float):
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array timestamps must not contain NaN values")
        if (timestamps < 0).any():
            raise ValueError("array timestamps must not contain negative values")
        if np.isnan(sampling_rate) or np.isinf(sampling_rate):
            raise ValueError("sampling_rate must not be NaN or infinite")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0")
        self.__timestamps = timestamps
        self.__sampling_rate = sampling_rate

    @property
    def start_time(self) -> float:
        # Event's start time in milliseconds
        return self.__timestamps[0]

    @property
    def end_time(self) -> float:
        # Event's end time in milliseconds
        return self.__timestamps[-1]

    @property
    def duration(self) -> float:
        # Event's duration in milliseconds
        return self.end_time - self.start_time

    @classmethod
    @abstractmethod
    def _event_type(cls) -> str:
        raise NotImplementedError

    @staticmethod
    def _split_samples_between_events(is_event: np.ndarray) -> List[np.ndarray]:
        """
        returns a list of arrays, each array contains the indices of the samples that belong to the same event
        """
        event_idxs = np.nonzero(is_event)[0]
        if len(event_idxs) == 0:
            return []
        event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
        different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
        return different_event_idxs

    def __repr__(self):
        return f"{self._event_type().capitalize()} ({self.duration:.1f} ms)"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, BaseEvent):
            return False
        if self.__sampling_rate != other.__sampling_rate:
            return False
        if self.__timestamps.shape != other.__timestamps.shape:
            return False
        if not np.allclose(self.__timestamps, other.__timestamps):
            return False
        return True
