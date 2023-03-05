import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from EventDetector.BaseDetector import BaseDetector


class BaseBlinkDetector(BaseDetector, ABC):
    """
    Baseclass for all blink event detectors
    """

    def __init__(self, missing_value: float = np.nan):
        self.__missing_value = missing_value

    @property
    def missing_value(self) -> float:
        return self.__missing_value


