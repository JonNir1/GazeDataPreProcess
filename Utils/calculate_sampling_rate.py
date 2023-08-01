import numpy as np
import pandas as pd
from typing import Union

import constants as cnst


def calculate_sampling_rate_from_milliseconds(milliseconds: np.ndarray) -> float:
    milliseconds = pd.Series(milliseconds)
    return cnst.MILLISECONDS_PER_SECOND / milliseconds.diff().mean()


def calculate_sampling_rate_from_microseconds(microseconds: np.ndarray) -> float:
    microseconds = pd.Series(microseconds)
    return cnst.MICROSECONDS_PER_SECOND / microseconds.diff().mean()


