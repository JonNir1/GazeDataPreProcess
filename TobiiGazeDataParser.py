import pandas as pd

from BaseGazeDataParser import BaseGazeDataParser


class TobiiGazeDataParser(BaseGazeDataParser):

    def __init__(self, path: str):
        super().__init__(path)

    @classmethod
    def MISSING_VALUE(cls) -> float:
        return -1

    @classmethod
    def TIME_COLUMN(cls) -> str:
        return 'RTTime'

    @classmethod
    def LEFT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXLeftEye'

    @classmethod
    def LEFT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYLeftEye'

    @classmethod
    def RIGHT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXRightEye'

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYRightEye'

    def parse_gaze_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep='\t')
        df = df.drop(columns=['LX', 'LY', 'RX', 'RY'])
        df = df.rename(columns={'Time': 'time', 'LX': 'left_x', 'LY': 'left_y', 'RX': 'right_x', 'RY': 'right_y'})
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _compute_sample_size_and_sr(self) -> (int, float):
        df = pd.read_csv(self.path, sep='\t')
        rt_time_micro = df['RTTimeMicro']
        num_samples = len(rt_time_micro)
        sampling_rate = 10**6 / rt_time_micro.diff().mode()
        return num_samples, sampling_rate
