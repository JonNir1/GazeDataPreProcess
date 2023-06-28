import numpy as np
from typing import List

from Config import experiment_config as cnfg
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector


class MissingDataBlinkDetector(BaseBlinkDetector):
    """
    Detects blinks from a single eye, defined as periods of missing data that last longer than min_duration.
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

    def detect_monocular(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects blinks in a single eye, defined as periods of missing data that last longer than min_duration.
        Based on implementation in https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L43
        :param x, y: 1D arrays of x- and y-coordinates, all in the same length

        :return: 1D array of booleans, True for blinks
        :raises ValueError: if timestamps, x and y are not of equal lengths
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        candidate_start_end_idxs = self.__find_blink_candidates(x, y)  # find blink candidates
        blink_start_end_idxs = [(start, end) for start, end in candidate_start_end_idxs if
                                end - start >= self._min_samples_within_event]  # exclude blinks that are too short

        # convert to boolean array
        is_blink = np.zeros(len(x), dtype=bool)
        if len(blink_start_end_idxs) == 0:
            return is_blink
        blink_idxs = np.concatenate([np.arange(start, end + 1) for start, end in blink_start_end_idxs])
        is_blink[blink_idxs] = True
        return is_blink

    def __find_blink_candidates(self, x: np.ndarray, y: np.ndarray) -> List[tuple]:
        """
        Detects periods of missing data and merges them together if they are close enough
        :param x, y: 1D arrays of x-coordinates and y-coordinates
        :return: list of tuples, each containing the start and end index of a blink candidate
        """
        # find idxs of missing data
        if self.missing_value is None or np.isnan(self.missing_value):
            is_missing = np.logical_or(np.isnan(x), np.isnan(y))
        else:
            is_missing = np.logical_or(x == self.missing_value, y == self.missing_value)
        missing_idxs = np.where(is_missing)[0]

        # find idxs of missing data that are close enough to merge together
        split_idxs = np.where(np.diff(missing_idxs) > self._min_samples_between_events)[0] + 1
        candidate_idxs_with_holes = np.split(missing_idxs, split_idxs)
        candidate_start_end = [(arr.min(), arr.max()) for arr in candidate_idxs_with_holes if arr.size > 0]
        return candidate_start_end
