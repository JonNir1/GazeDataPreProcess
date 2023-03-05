import pandas as pd

from EventDetector.BaseGazeEventDetector import BaseGazeEventDetector


class EngbertGazeEventDetector(BaseGazeEventDetector):

    def __init__(self, gaze_data: pd.DataFrame, sampling_rate: int):
        return None

    def detect_blinks(self) -> pd.DataFrame:
        pass

    def detect_saccades(self) -> pd.DataFrame:
        pass

    def detect_fixations(self) -> pd.DataFrame:
        pass

    def detect_smooth_pursuit(self) -> pd.DataFrame:
        raise NotImplementedError

