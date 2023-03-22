import numpy as np
import pandas as pd
from typing import List, Tuple

import experiment_config as conf
from EventDetectors.BaseSaccadeDetector import BaseSaccadeDetector

DEFAULT_DERIVATION_WINDOW_SIZE = 3
DEFAULT_LAMBDA_NOISE_THRESHOLD = 5


class EngbertSaccadeDetector(BaseSaccadeDetector):
    """
    Saccade detector based on the algorithm described by Engbert, Kliegl, and Mergenthaler (2003, 2006).
    See implementations in the following repositories:
        - https://github.com/Yuvishap/Gaze-Project/blob/67acb26fc90e5148a05b47ca7711306c94b79ed7/Gaze/src/pre_processing/business/EngbertFixationsLogic.py
        - https://github.com/odedwer/EyelinkProcessor/blob/66f56463ba8d2ad75f7935e3d020b051fb2aa4a4/SaccadeDetectors.py
        - https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L175
    """

    def __init__(self,
                 min_duration: float = conf.SACCADE_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = conf.INTER_EVENT_TIME,
                 derivation_window_size: int = DEFAULT_DERIVATION_WINDOW_SIZE,
                 lambda_noise_threshold: int = DEFAULT_LAMBDA_NOISE_THRESHOLD):
        super().__init__(min_duration, sr, iet)
        self.__derivation_window_size = derivation_window_size
        self.__lambda_noise_threshold = lambda_noise_threshold

    def detect(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        is_saccade_candidate = self._find_candidates(x, y)
        saccades_start_end_idxs = self._find_start_end_indices(is_saccade_candidate)

        # convert to boolean array
        saccade_idxs = np.concatenate([np.arange(start, end + 1) for start, end in saccades_start_end_idxs])
        is_saccade = np.zeros(len(x), dtype=bool)
        is_saccade[saccade_idxs] = True
        return is_saccade

    def _find_candidates(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects saccade candidates of a single eye, in the given gaze data.
        A saccade candidate is a sample that has a velocity greater than the noise threshold, calculated as the multiple
            of the velocity's median-standard-deviation with the constant self.__lambda_noise_threshold.
        :param x: gaze positions on the x-axis
        :param y: gaze positions on the y-axis

        :return: boolean array of the same length as the given data, indicating whether a sample is a saccade candidate

        :raises ValueError: if the given data is not of the same length
        :raises ValueError: if the given data is not of length at least 2 * self.derivation_window_size
        """
        if len(x) != len(y):
            raise ValueError("x and y must be of the same length")
        if len(x) < 2 * self.__derivation_window_size:
            raise ValueError(
                f"x and y must be of length at least 2 * derivation_window_size (={2 * self.__derivation_window_size})")

        vel_x = self.__numerical_derivative(x, n=self.__derivation_window_size)
        sd_x = self.__median_standard_deviation(vel_x)
        vel_y = self.__numerical_derivative(y, n=self.__derivation_window_size)
        sd_y = self.__median_standard_deviation(vel_y)

        ellipse_thresholds = np.power(vel_x / (sd_x * self.__lambda_noise_threshold), 2) + np.power(
            vel_y / (sd_y * self.__lambda_noise_threshold), 2)
        is_saccade_candidate = ellipse_thresholds > 1
        return is_saccade_candidate.values

    def _find_start_end_indices(self, is_saccade_candidate: np.ndarray) -> List[Tuple[int, int]]:
        """
        Excludes saccade candidates that are shorter than the minimum duration of a saccade.
        :param is_saccade_candidate: boolean array indicating whether a sample is a saccade candidate
        :return: list of tuples, each tuple containing the start and end indices of a saccade
        """
        # split saccade candidates to separate saccades
        saccade_candidate_idxs = np.nonzero(is_saccade_candidate)[0]
        splitting_idxs = np.where(np.diff(saccade_candidate_idxs) > self._min_samples_between_events)[
                             0] + 1  # +1 because we want the index after the split
        separate_saccade_idxs = np.split(saccade_candidate_idxs, splitting_idxs)

        # exclude saccades that are shorter than the minimum duration
        saccades_start_end = list(map(lambda sac_idxs: (sac_idxs.min(), sac_idxs.max()), separate_saccade_idxs))
        saccades_start_end = list(
            filter(lambda sac: sac[1] - sac[0] >= self._min_samples_within_event, saccades_start_end))
        return saccades_start_end

    @staticmethod
    def __numerical_derivative(v, n: int) -> np.ndarray:
        """
        Calculates the numerical derivative of the given values, as described by Engbert & Kliegl(2003):
            dX/dt = [(v[X+(N-1)] + v[X+(N-2)] + ... + v[X+1]) - (v[X-(N-1)] + v[X-(N-2)] + ... + v[X-1])] / 2N

        :param v: series of length N to calculate the derivative for
        :param n: number of samples to use for the calculation
        :return: numerical derivative of the given values
                Note: the first and last (n-1) samples will be NaN
        """
        N = len(v)
        if n <= 0:
            raise ValueError("n must be greater than 0")
        if n >= int(0.5 * N):
            raise ValueError("n must be less than half the length of the given values")
        if not isinstance(v, pd.Series):
            # convert to pd series to use rolling window function
            v = pd.Series(v)
        v[v < 0] = np.nan
        prev_elements_sum = v.rolling(n - 1).sum().shift(1)
        next_elements_sum = v.rolling(n - 1).sum().shift(1 - n)
        deriv = (next_elements_sum - prev_elements_sum) / (2 * n)
        return deriv

    @staticmethod
    def __median_standard_deviation(v: np.ndarray, min_sd: float = 1e-6) -> float:
        """
        Calculates the median-based standard deviation of the given values.
        :param v: values to calculate the median standard deviation for
        :param min_sd: minimum standard deviation to return
        :return: median standard deviation
        """
        assert min_sd > 0, "min_sd must be greater than 0"
        squared_median = np.power(np.nanmedian(v), 2)
        median_of_squared = np.nanmedian(np.power(v, 2))
        sd = np.sqrt(median_of_squared - squared_median)
        return max(sd, min_sd)
