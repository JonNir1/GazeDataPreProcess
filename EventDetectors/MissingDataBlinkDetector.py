import numpy as np
from typing import List

from Config import experiment_config as cnfg
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector


class MissingDataBlinkDetector(BaseBlinkDetector):
    """
    Detects blinks from a single eye, defined as periods of missing data that last longer than min_duration.
    See implementation in https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L43

    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    - missing_value: default value indicating missing data                  (default: np.nan)
    - inter_event_time: minimal time between two (same) events in ms        (default: 5)
    """

    def __init__(self,
                 sr: float,
                 missing_value: float = cnfg.DEFAULT_MISSING_VALUE,
                 min_duration: float = cnfg.DEFAULT_BLINK_MINIMUM_DURATION,
                 iet: float = cnfg.DEFAULT_INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
        self.__missing_value = missing_value

    @property
    def missing_value(self) -> float:
        return self.__missing_value

    def set_missing_value(self, missing_value: float):
        self.__missing_value = missing_value

    def _find_candidates(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array of the same length as the input data, where True indicates the samples that are
        candidates for being part of a blink. A blink candidate is defined as a sample with missing x- or y-coordinates.

        :param x: gaze positions on the x-axis
        :param y: gaze positions on the y-axis
        :return: boolean array of the same length as the given data, indicating whether a sample is a blink candidate
        """
        if self.missing_value is None or np.isnan(self.missing_value):
            return np.logical_or(np.isnan(x), np.isnan(y))
        return np.logical_or(x == self.missing_value, y == self.missing_value)
