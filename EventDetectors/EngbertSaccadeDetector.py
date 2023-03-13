import numpy as np
import pandas as pd

import EventDetectors.utils as u
from EventDetectors.BaseSaccadeDetector import BaseSaccadeDetector


class EngbertSaccadeDetector(BaseSaccadeDetector):
    """
    Saccade detector based on the algorithm described by Engbert, Kliegl, and Mergenthaler (2003, 2006).
    See implementations in the following repositories:
        - https://github.com/Yuvishap/Gaze-Project/blob/67acb26fc90e5148a05b47ca7711306c94b79ed7/Gaze/src/pre_processing/business/EngbertFixationsLogic.py
        - https://github.com/odedwer/EyelinkProcessor/blob/66f56463ba8d2ad75f7935e3d020b051fb2aa4a4/SaccadeDetectors.py
        - https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L175
    """

    DERIVATION_WINDOW_SIZE = 3
    LAMBDA_NOISE_THRESHOLD = 5

    def detect(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects saccades of a single eye, in the given gaze data.
        :param x:
        :param y:
        :return:
        """
        is_saccade_candidate = self._detect_saccade_candidates(x, y)

        raise NotImplementedError

    @classmethod
    def __find_saccade_candidates(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects saccade candidates of a single eye, in the given gaze data.
        A saccade candidate is a sample that has a velocity greater than the noise threshold, calculated as the multiple
            of the velocity's median-standard-deviation with the constant LAMBDA_NOISE_THRESHOLD.
        :param x: gaze positions on the x-axis
        :param y: gaze positions on the y-axis

        :return: boolean array of the same length as the given data, indicating whether a sample is a saccade candidate

        :raises ValueError: if the given data is not of the same length
        :raises ValueError: if the given data is not of length at least 2 * DERIVATION_WINDOW_SIZE
        """
        if len(x) != len(y):
            raise ValueError("x and y must be of the same length")
        if len(x) < 2 * cls.DERIVATION_WINDOW_SIZE:
            raise ValueError(f"x and y must be of length at least 2 * DERIVATION_WINDOW_SIZE (={2 * cls.DERIVATION_WINDOW_SIZE})")

        vel_x = u.numerical_derivative(x, n=cls.DERIVATION_WINDOW_SIZE)
        sd_x = u.median_standard_deviation(vel_x)
        vel_y = u.numerical_derivative(y, n=cls.DERIVATION_WINDOW_SIZE)
        sd_y = u.median_standard_deviation(vel_y)

        ellipse_thresholds = np.power(vel_x / (sd_x * cls.LAMBDA_NOISE_THRESHOLD), 2) + np.power(vel_y / (sd_y * cls.LAMBDA_NOISE_THRESHOLD), 2)
        is_saccade_candidate = ellipse_thresholds > 1
        return is_saccade_candidate.values




