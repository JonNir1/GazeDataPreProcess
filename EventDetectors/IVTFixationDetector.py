import numpy as np

import EventDetectors.scripts.event_detector_utils as u
from EventDetectors.BaseFixationDetector import BaseFixationDetector, DEFAULT_FIXATION_MINIMUM_DURATION

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

    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - velocity_threshold: velocity threshold in degrees per second          (default: 20)
    - min_duration: minimum duration of a blink in milliseconds             (default: 55)
    - inter_event_time: minimal time between two (same) events in ms        (default: 5)
    """

    def __init__(self,
                 sr: float,
                 velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
                 min_duration: float = DEFAULT_FIXATION_MINIMUM_DURATION,
                 iet: float = BaseFixationDetector.DEFAULT_INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
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


