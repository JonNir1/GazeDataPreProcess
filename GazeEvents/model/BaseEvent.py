import numpy as np
from abc import ABC, abstractmethod


class BaseEvent(ABC):

    def __init__(self, timestamps: np.ndarray):
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array timestamps must not contain NaN values")
        if (timestamps < 0).any():
            raise ValueError("array timestamps must not contain negative values")
        self.__timestamps = timestamps

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
