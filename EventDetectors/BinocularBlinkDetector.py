import numpy as np

import experiment_config as conf
import constants as cnst
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector, DEFAULT_MINIMUM_DURATION
from EventDetectors.MonocularBlinkDetector import MonocularBlinkDetector


class BinocularBlinkDetector(BaseBlinkDetector):
    """
    Detects blinks from both eyes, using two MonocularBlinkDetectors objects.
    Define a criterion for merging the results of the two detectors:
    - "AND": a blink is detected if both eyes are missing data for a period longer than min_duration
    - "OR": a blink is detected if at least one eye is missing data for a period longer than min_duration
    """

    def __init__(self,
                 criterion: str = "OR",
                 min_duration: float = DEFAULT_MINIMUM_DURATION,
                 time_between_blinks: float = 20,
                 missing_value: float = cnst.MISSING_VALUE,
                 sr: float = conf.SAMPLING_RATE):
        super().__init__(min_duration=min_duration,
                         time_between_blinks=time_between_blinks,
                         missing_value=missing_value,
                         sr=sr)
        if criterion.upper() not in ["AND", "OR"]:
            raise ValueError("criterion must be either 'AND' or 'OR'")
        self.__criterion = criterion
        self.__left_detector = MonocularBlinkDetector(min_duration=min_duration,
                                                      time_between_blinks=time_between_blinks,
                                                      missing_value=missing_value,
                                                      sr=sr)
        self.__right_detector = MonocularBlinkDetector(min_duration=min_duration,
                                                       time_between_blinks=time_between_blinks,
                                                       missing_value=missing_value,
                                                       sr=sr)

    def detect(self,
               left_x: np.ndarray, left_y: np.ndarray,
               right_x: np.ndarray, right_y: np.ndarray) -> np.ndarray:
        if len(left_x) != len(left_y):
            raise ValueError("left_x and left_y must have the same length")
        if len(right_x) != len(right_y):
            raise ValueError("right_x and right_y must have the same length")

        left_blinks = self.__left_detector.detect(left_x, left_y)
        right_blinks = self.__right_detector.detect(right_x, right_y)
        if self.criterion.upper() == "AND":
            return np.logical_and(left_blinks, right_blinks)
        return np.logical_or(left_blinks, right_blinks)

    @property
    def criterion(self) -> str:
        return self.__criterion

    def set_criterion(self, criterion: str):
        if criterion.upper() not in ["AND", "OR"]:
            raise ValueError("criterion must be either 'AND' or 'OR'")
        self.__criterion = criterion

    def set_min_duration(self, min_duration: float):
        self.__min_duration = min_duration
        self.__left_detector.set_min_duration(min_duration)
        self.__right_detector.set_min_duration(min_duration)

    def set_missing_value(self, missing_value: float):
        self.__missing_value = missing_value
        self.__left_detector.set_missing_value(missing_value)
        self.__right_detector.set_missing_value(missing_value)

    def set_sampling_rate(self, sampling_rate: float):
        self.__sampling_rate = sampling_rate
        self.__left_detector.set_sampling_rate(sampling_rate)
        self.__right_detector.set_sampling_rate(sampling_rate)

