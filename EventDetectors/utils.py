from math import ceil
import numpy as np
import pandas as pd

import constants as c


def calculate_minimum_sample_count(min_duration: float, sampling_rate: float) -> int:
    """
    Calculates the minimum number of consecutive samples that must be present to be considered an event.
    :param min_duration: minimum duration of an event in milliseconds
    :param sampling_rate: sampling rate of the data in Hz
    :return: minimum number of consecutive samples
    """
    return ceil(min_duration * sampling_rate / c.MILLISECONDS_PER_SECOND)


def numerical_derivative(v, n: int) -> np.ndarray:
    """
    Calculates the numerical derivative of the given values, as described by Engbert & Kliegl(2003):
        dX/dt = [(v[X+(N-1)] + v[X+(N-2)] + ... + v[X+1]) - (v[X-(N-1)] + v[X-(N-2)] + ... + v[X-1])] / 2N

    :param v: series of length N to calculate the derivative for
    :param n: number of samples to use for the calculation
    :return: numerical derivative of the given values
            Note: the first and last (n-1) samples will be NaN
    """
    N = len(v)
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if n >= int(0.5 * N):
        raise ValueError("n must be less than half the length of the given values")
    if not isinstance(v, pd.Series):
        v = pd.Series(v)
    v[v < 0] = np.nan
    prev_elements_sum = v.rolling(n - 1).sum().shift(1)
    next_elements_sum = v.rolling(n - 1).sum().shift(1 - n)
    deriv = (next_elements_sum - prev_elements_sum) / (2 * n)
    return deriv


def median_standard_deviation(v: np.ndarray, min_sd: float = 1e-6) -> float:
    """
    Calculates the median-based standard deviation of the given values.
    :param v: values to calculate the median standard deviation for
    :param min_sd: minimum standard deviation to return
    :return: median standard deviation
    """
    assert min_sd > 0, "min_sd must be greater than 0"
    squared_median = np.power(np.nanmedian(v), 2)
    median_of_squared = np.nanmedian(np.power(v, 2))
    sd = np.sqrt(median_of_squared - squared_median)
    return max(sd, min_sd)
