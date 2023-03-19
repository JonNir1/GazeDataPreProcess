import os
import numpy as np
import pandas as pd
from typing import List

import constants as cnst
from DataParser.BaseParser import BaseParser


class TriggerLogParser(BaseParser):
    TIME_COLUMN = 'ClockTime'
    TRIGGER_COLUMN = 'BioSemiCode'

    def __init__(self, input_path: str, start_trigger: int, end_trigger: int):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'File not found: {input_path}')
        self.input_path = input_path
        self.start_trigger = start_trigger
        self.end_trigger = end_trigger

    def parse(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path, sep='\t')
        df = df[self.get_columns()]
        df.rename(columns=lambda col: self._column_name_mapper(col), inplace=True)
        return df

    def parse_and_split(self) -> List[pd.DataFrame]:
        full_df = self.parse()
        start_idxs = np.nonzero(full_df[cnst.TRIGGER] == self.start_trigger)[0]
        end_idxs = np.nonzero(full_df[cnst.TRIGGER] == self.end_trigger)[0]
        if len(start_idxs) != len(end_idxs):
            raise AssertionError(f'Number of start triggers ({len(start_idxs)}) does not match number of end triggers ({len(end_idxs)})')
        start_end_idxs = np.vstack([start_idxs, end_idxs]).T
        df_list = [full_df.iloc[start:end + 1] for start, end in start_end_idxs]
        for i, sub_df in enumerate(df_list):
            sub_df[cnst.TRIAL] = i+1
        return df_list

    @classmethod
    def get_columns(cls) -> List[str]:
        return [cls.TIME_COLUMN, cls.TRIGGER_COLUMN]

    @classmethod
    def MILLISECONDS_COLUMN(cls) -> str:
        return cls.TIME_COLUMN

    @classmethod
    def _column_name_mapper(cls, column_name: str) -> str:
        if column_name == cls.TIME_COLUMN:
            return cnst.MILLISECONDS
        if column_name == cls.TRIGGER_COLUMN:
            return cnst.TRIGGER
        raise ValueError(f'No name-mapping for column {column_name} in TriggerParser')


