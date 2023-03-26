import numpy as np

import experiment_config as conf
import EventDetectors.utils as u
from EventDetectors.BaseFixationDetector import BaseFixationDetector

DEFAULT_VELOCITY_THRESHOLD = 20  # degrees per second


class IVTFixationDetector(BaseFixationDetector):
    """
    Detects fixations using the algorithm described by Salvucci & Goldberg (2000): "Identifying fixations and saccades
        in eye-tracking protocols".

    This Detector determines if a sample is a fixation by checking if the velocity of the sample
        is below a predefined velocity threshold.
    In the original paper they used a threshold of 20 degrees per second. In a review paper by Andersson et al. ("One
        algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection algorithms"; 2016),
        they used a threshold of 45 degrees per second.

    See original implementation: https://github.com/ecekt/eyegaze/blob/master/gaze.py
    """

    def __init__(self,
                 velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
                 min_duration: float = BaseFixationDetector.FIXATION_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = BaseFixationDetector.INTER_EVENT_TIME):
        super().__init__(min_duration, sr, iet)
        self.__velocity_threshold = velocity_threshold

    @property
    def velocity_threshold(self) -> float:
        return self.__velocity_threshold

    def set_velocity_threshold(self, velocity_threshold: float):
        self.__velocity_threshold = velocity_threshold

    def detect(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects fixations in the given data.

        :param x: x-coordinates of the gaze data.
        :param y: y-coordinates of the gaze data.
        :return: A boolean array of the same length as the input data, where True indicates a fixation.
        """
        velocities = self.__calculate_angular_velocity(x, y)
        is_fixation_candidate = velocities <= self.velocity_threshold
        # TODO: merge close candidates and filter out short ones
        raise NotImplementedError

    def __calculate_angular_velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the angular velocity of the gaze data.
        :param x: 1D array of x-coordinates.
        :param y: 1D array of y-coordinates.
        :return: 1D array of angular velocities.
        """
        x_shifted = u.shift_array(x, 1)
        y_shifted = u.shift_array(y, 1)
        pixels = np.vstack([x, y, x_shifted, y_shifted])  # shape (4, N)
        pixels2D = np.array([pixels[:, i].reshape(2, 2) for i in range(pixels.shape[1])])  # shape (N, 2, 2)
        angles = np.array([u.pixels2deg(pixels2D[i]) for i in range(pixels2D.shape[0])])
        return angles * self.sampling_rate


