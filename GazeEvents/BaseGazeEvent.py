from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, final

import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class BaseGazeEvent(ABC):
    _EVENT_TYPE: GazeEventTypeEnum
    MIN_DURATION: float = 1      # minimum duration of an event in milliseconds
    MAX_DURATION: float = 10000  # maximum duration of an event in milliseconds

    def __init__(self, timestamps: np.ndarray):
        # set instance attributes:
        if len(timestamps) < cnfg.DEFAULT_MINIMUM_SAMPLES_PER_EVENT:
            raise ValueError("event must be at least {} samples long".format(cnfg.DEFAULT_MINIMUM_SAMPLES_PER_EVENT))
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array timestamps must not contain NaN values")
        if (timestamps < 0).any():
            raise ValueError("array timestamps must not contain negative values")
        self._timestamps = timestamps

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of event information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
        """
        return pd.Series(data=[self.event_type(), self.start_time, self.end_time, self.duration, self.is_outlier],
                         index=["event_type", "start_time", "end_time", "duration", "is_outlier"])

    @final
    @property
    def start_time(self) -> float:
        # Event's start time in milliseconds
        return self._timestamps[0]

    @final
    @property
    def end_time(self) -> float:
        # Event's end time in milliseconds
        return self._timestamps[-1]

    @final
    @property
    def duration(self) -> float:
        # Event's duration in milliseconds
        return self.end_time - self.start_time

    @final
    @property
    def is_outlier(self) -> bool:
        return len(self.get_outlier_reasons()) > 0

    def get_outlier_reasons(self) -> List[str]:
        reasons = []
        if self.duration < self.MIN_DURATION:
            reasons.append(f"min_{cnst.DURATION}")
        if self.duration > self.MAX_DURATION:
            reasons.append(f"max_{cnst.DURATION}")
        return reasons

    @classmethod
    @final
    def event_type(cls) -> str:
        return cls._EVENT_TYPE.name.lower()

    @classmethod
    @final
    def set_min_duration(cls, min_duration: float):
        cls.MIN_DURATION = min_duration

    @classmethod
    @final
    def set_max_duration(cls, max_duration: float):
        cls.MAX_DURATION = max_duration

    def __repr__(self):
        return f"{self.event_type().capitalize()} ({self.duration:.1f} ms)"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self._timestamps.shape != other._timestamps.shape:
            return False
        if not np.allclose(self._timestamps, other._timestamps):
            return False
        return True
