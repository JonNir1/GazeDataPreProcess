import pandas as pd
from abc import ABC, abstractmethod


class BaseGazeEventDetector(ABC):

    @abstractmethod
    def detect_blinks(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def detect_saccades(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def detect_fixations(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def detect_smooth_pursuit(self) -> pd.DataFrame:
        raise NotImplementedError

