import numpy as np
import pandas as pd
from typing import List

import experiment_config as conf
import constants as cnst
from DataParser.BaseGazeDataParser import BaseGazeDataParser


class TobiiGazeDataParser(BaseGazeDataParser):

    def __init__(self, path: str):
        super().__init__(path)

    def parse(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep='\t')
        df.drop(columns=[col for col in df.columns if col not in self.get_columns()], inplace=True)
        df.replace(to_replace=self.MISSING_VALUE(), value=np.nan, inplace=True)

        # correct for screen resolution
        df[self.LEFT_X_COLUMN()] = df[self.LEFT_X_COLUMN()] * conf.SCREEN_WIDTH
        df[self.LEFT_Y_COLUMN()] = df[self.LEFT_Y_COLUMN()] * conf.SCREEN_HEIGHT
        df[self.RIGHT_X_COLUMN()] = df[self.RIGHT_X_COLUMN()] * conf.SCREEN_WIDTH
        df[self.RIGHT_Y_COLUMN()] = df[self.RIGHT_Y_COLUMN()] * conf.SCREEN_HEIGHT

        # reorder + rename columns to match the standard
        df = df[self.get_columns()]
        df.rename(columns=lambda col: self._column_name_mapper(col), inplace=True)
        return df

    @classmethod
    def MISSING_VALUE(cls) -> float:
        return -1

    @classmethod
    def TRIAL_COLUMN(cls) -> str:
        return 'RunningSample'

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
    def LEFT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterLeftEye"

    @classmethod
    def RIGHT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXRightEye'

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYRightEye'

    @classmethod
    def RIGHT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterRightEye"

    @classmethod
    def ADDITIONAL_COLUMNS(cls) -> List[str]:
        return conf.ADDITIONAL_COLUMNS

    def _compute_sample_size_and_sr(self) -> (int, float):
        df = pd.read_csv(self.path, sep='\t')
        rt_time_micro = df['RTTimeMicro']
        num_samples = len(rt_time_micro)
        sampling_rate = cnst.MICROSECONDS_PER_SECOND / rt_time_micro.diff().mode()
        return num_samples, sampling_rate

    @classmethod
    def _column_name_mapper(cls, column_name: str) -> str:
        if column_name in cls._get_common_columns():
            return super()._column_name_mapper(column_name)
        if column_name in cls.ADDITIONAL_COLUMNS():
            return column_name
        raise ValueError(f'No name-mapping for column {column_name}')
