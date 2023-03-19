import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class BaseParser(ABC):

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        # parse the data and return a pandas DataFrame
        raise NotImplementedError

    @abstractmethod
    def parse_and_split(self) -> List[pd.DataFrame]:
        # parse the data and return a list of pandas DataFrames, one for each trial
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_columns(cls) -> List[str]:
        # list of column names to parse
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MILLISECONDS_COLUMN(cls) -> str:
        # column name for time in milliseconds
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _column_name_mapper(cls, column_name: str) -> str:
        raise NotImplementedError
