import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseParser(ABC):

    @abstractmethod
    def parse(self, input_path: str, additional_columns: Optional[List[str]] = None,
              output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Reads the input file and parses it into a pandas DataFrame with a standardized set of columns, while including
        any additional columns specified in the additional_columns parameter. If the output_path parameter is specified,
        the parsed DataFrame is also saved to the specified path.

        :param input_path: the path to the input file
        :param additional_columns: list of additional columns to include in the output DataFrame (optional)
        :param output_path: the path to the output file (optional)

        :raises FileNotFoundError: if the input file does not exist
        """
        # parse the data and return a pandas DataFrame
        raise NotImplementedError

    @abstractmethod
    def parse_and_split(self, input_path: str, additional_columns: Optional[List[str]] = None,
                        output_path: Optional[str] = None) -> List[pd.DataFrame]:
        """
        Reads the input file and parses it into a list of pandas DataFrames, one for each trial, with a standardized
        set of columns, while including any additional columns specified in the additional_columns parameter.
        If the output_path parameter is specified, the parsed DataFrame is also saved to the specified path.

        :param input_path: the path to the input file
        :param additional_columns: list of additional columns to include in the output DataFrames (optional)
        :param output_path: the path to the output file (optional)

        :raises FileNotFoundError: if the input file does not exist
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_common_columns(cls) -> List[str]:
        # list of column names that are common for all parsers
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _column_name_mapper(cls, column_name: str) -> str:
        # maps the input column name to a new (standardized) column name
        raise NotImplementedError
