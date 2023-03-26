import numpy as np
from abc import ABC, abstractmethod


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
