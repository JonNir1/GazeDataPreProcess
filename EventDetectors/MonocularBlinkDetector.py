import numpy as np
from typing import List

import EventDetectors.utils as u
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector


class MonocularBlinkDetector(BaseBlinkDetector):
    """
    Detects blinks from a single eye.
    """

    def detect(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects blinks in a single eye, defined as periods of missing data that last longer than min_duration.
        Based on implementation in https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L43
        :param timestamps, x, y: 1D arrays of timestamps, x-coordinates and y-coordinates, all in the same length

        :return: 1D array of booleans, True for blinks
        :raises ValueError: if timestamps, x and y are not of equal lengths
        """
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("timestamps, x and y must have the same length")
        sr = u.calculate_sampling_rate(timestamps)

        # find blink candidates
        max_length_between_candidates = u.calculate_minimum_sample_count(self.time_between_blinks, sr)
        candidate_start_end_idxs = self.__find_blink_candidates(x, y, max_length_between_candidates)

        # exclude blinks that are too short
        min_length_for_blink = u.calculate_minimum_sample_count(self.min_duration, sr)
        blink_start_end_idxs = [(start, end) for start, end in candidate_start_end_idxs if
                                end - start >= min_length_for_blink]

        # convert to boolean array
        blink_idxs = np.concatenate([np.arange(start, end + 1) for start, end in blink_start_end_idxs])
        is_blink = np.zeros(len(timestamps), dtype=bool)
        is_blink[blink_idxs] = True
        return is_blink

    def __find_blink_candidates(self, x: np.ndarray, y: np.ndarray, merge_threshold: int) -> List[tuple]:
        """
        Detects periods of missing data and merges them together if they are close enough
        :param x, y: 1D arrays of x-coordinates and y-coordinates
        :param merge_threshold: maximum number of samples between two missing data periods to merge them together
        :return: list of tuples, each containing the start and end index of a blink candidate
        """
        assert merge_threshold >= 0, "merge_threshold must be non-negative"

        # find idxs of missing data
        is_missing = np.logical_or(x == self.missing_value, y == self.missing_value)
        missing_idxs = np.where(is_missing)[0]

        # find idxs of missing data that are close enough to merge together
        split_idxs = np.where(np.diff(missing_idxs) > merge_threshold)[0] + 1
        candidate_idxs_with_holes = np.split(is_missing, split_idxs)
        candidate_start_end = [(arr.min(), arr.max()) for arr in candidate_idxs_with_holes]
        return candidate_start_end

