import numpy as np
import pandas as pd


def shift_array(array: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts an array by a given amount.
        If the shift is positive, the array is shifted to the right.
        If the shift is negative, the array is shifted to the left.
    :param array: np.ndarray - the array to be shifted.
    :param shift: int - the amount to shift the array by.
    :return: shifted_array: np.ndarray - the shifted array.
    """
    shifted_array = np.roll(array, shift)
    if shift > 0:
        shifted_array[:shift] = np.nan
    elif shift < 0:
        shifted_array[shift:] = np.nan
    return shifted_array


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
        # convert to pd series to use rolling window function
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