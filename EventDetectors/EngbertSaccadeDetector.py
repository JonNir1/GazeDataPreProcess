import numpy as np
from typing import List, Tuple

from Config import experiment_config as cnfg
import Utils.array_utils as au
from EventDetectors.BaseSaccadeDetector import BaseSaccadeDetector

DEFAULT_DERIVATION_WINDOW_SIZE = 3
DEFAULT_LAMBDA_NOISE_THRESHOLD = 5


class EngbertSaccadeDetector(BaseSaccadeDetector):
    """
    Saccade detector based on the algorithm described by Engbert, Kliegl, and Mergenthaler (2003, 2006).
    See implementations in the following repositories:
        - https://github.com/Yuvishap/Gaze-Project/blob/master/Gaze/src/pre_processing/business/EngbertFixationsLogic.py
        - https://github.com/odedwer/EyelinkProcessor/blob/master/SaccadeDetectors.py
        - https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py

    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - min_duration: minimum duration of a saccade in milliseconds                     (default: 5)
    - inter_event_time: minimal time between two (same) events in ms                (default: 5)
    - derivation_window_size: size of the window used to calculate the derivative   (default: 3)
    - lambda_noise_threshold: threshold for the lambda noise value                  (default: 5)
    """

    def __init__(self,
                 sr: float,
                 min_duration: float = cnfg.DEFAULT_SACCADE_MINIMUM_DURATION,
                 iet: float = cnfg.DEFAULT_INTER_EVENT_TIME,
                 derivation_window_size: int = DEFAULT_DERIVATION_WINDOW_SIZE,
                 lambda_noise_threshold: int = DEFAULT_LAMBDA_NOISE_THRESHOLD):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
        self.__derivation_window_size = derivation_window_size
        self.__lambda_noise_threshold = lambda_noise_threshold

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

        vel_x = au.numerical_derivative(x, n=self.__derivation_window_size)
        sd_x = au.median_standard_deviation(vel_x)
        vel_y = au.numerical_derivative(y, n=self.__derivation_window_size)
        sd_y = au.median_standard_deviation(vel_y)
        ellipse_thresholds = np.power(vel_x / (sd_x * self.__lambda_noise_threshold), 2) + np.power(
            vel_y / (sd_y * self.__lambda_noise_threshold), 2)
        is_saccade_candidate = ellipse_thresholds > 1
        return is_saccade_candidate.values
