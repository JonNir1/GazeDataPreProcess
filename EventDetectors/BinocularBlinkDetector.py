import numpy as np

import constants as cnst
from EventDetectors.BaseBlinkDetector import BaseBlinkDetector
from EventDetectors.MonocularBlinkDetector import MonocularBlinkDetector


class BinocularBlinkDetector(BaseBlinkDetector):
    """
    Detects blinks from both eyes, using two MonocularBlinkDetectors objects.
    Define a criterion for merging the results of the two detectors:
    - "AND": a blink is detected if both eyes are missing data for a period longer than min_duration
    - "OR": a blink is detected if at least one eye is missing data for a period longer than min_duration
    """

    def __init__(self,
                 time_between_blinks: float = 20,
                 min_duration: float = 50,
                 criterion: str = "OR"):
        super().__init__(min_duration=min_duration, time_between_blinks=time_between_blinks)
        if criterion.upper() not in ["AND", "OR"]:
            raise ValueError("criterion must be either 'AND' or 'OR'")
        self.__criterion = criterion

    def detect(self,
               left_x: np.ndarray, left_y: np.ndarray,
               right_x: np.ndarray, right_y: np.ndarray) -> np.ndarray:
        if len(left_x) != len(left_y):
            raise ValueError("left_x and left_y must have the same length")
        if len(right_x) != len(right_y):
            raise ValueError("right_x and right_y must have the same length")
        left_detector = MonocularBlinkDetector(min_duration=self.min_duration,
                                               time_between_blinks=self.time_between_blinks)
        right_detector = MonocularBlinkDetector(min_duration=self.min_duration,
                                                time_between_blinks=self.time_between_blinks)

        left_blinks = left_detector.detect(left_x, left_y)
        right_blinks = right_detector.detect(right_x, right_y)

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

