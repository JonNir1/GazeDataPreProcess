import numpy as np

import experiment_config as cnfg
from Utils import angle_utils as angle_utils
from EventDetectors.BaseFixationDetector import BaseFixationDetector


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
                 velocity_threshold: float = cnfg.DEFAULT_FIXATION_MAX_VELOCITY,
                 min_duration: float = cnfg.DEFAULT_FIXATION_MINIMUM_DURATION,
                 iet: float = cnfg.DEFAULT_INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
        self.__velocity_threshold = velocity_threshold

    @property
    def velocity_threshold(self) -> float:
        return self.__velocity_threshold

    def set_velocity_threshold(self, velocity_threshold: float):
        self.__velocity_threshold = velocity_threshold

    def detect_monocular(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Detects fixations in the given data.

        :param x: x-coordinates of the gaze data.
        :param y: y-coordinates of the gaze data.
        :return: A boolean array of the same length as the input data, where True indicates a fixation.
        """
        # velocities = vu.calculate_angular_velocity(x, y, self.sampling_rate)
        # is_fixation_candidate = velocities <= self.velocity_threshold
        # TODO: merge close candidates and filter out short ones
        raise NotImplementedError


