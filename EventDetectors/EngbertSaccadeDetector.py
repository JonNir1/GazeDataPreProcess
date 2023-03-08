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
        if len(x) != len(y):
            raise ValueError("x and y must be of the same length")
        vel_x = u.numerical_derivative(x, n=self.DERIVATION_WINDOW_SIZE)
        vel_y = u.numerical_derivative(y, n=self.DERIVATION_WINDOW_SIZE)
        raise NotImplementedError


