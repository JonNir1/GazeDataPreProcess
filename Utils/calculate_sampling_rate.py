import numpy as np
import pandas as pd
from typing import Union

import constants as cnst


def calculate_sampling_rate_from_milliseconds(milliseconds: Union[np.ndarray, pd.Series]) -> float:
    if isinstance(milliseconds, np.ndarray):
        milliseconds = pd.Series(milliseconds)
    return cnst.MILLISECONDS_PER_SECOND / milliseconds.diff().mean()


def calculate_sampling_rate_from_microseconds(microseconds: Union[np.ndarray, pd.Series]) -> float:
    if isinstance(microseconds, np.ndarray):
        microseconds = pd.Series(microseconds)
    return cnst.MICROSECONDS_PER_SECOND / microseconds.diff().mean()


