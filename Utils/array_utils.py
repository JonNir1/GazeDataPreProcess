import numpy as np
import pandas as pd
from typing import List


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


def numerical_derivative(x, n: int) -> np.ndarray:
    """
    Calculates the numerical derivative of the given values, as described by Engbert & Kliegl(2003):
        dx/dt = [(x[t+(n-1)] + x[t+(n-2)] + ... + x[t+1]) - (x[t-(n-1)] + x[t-(n-2)] + ... + X[t-1])] / 2n

    :param x: series of length N to calculate the derivative for
    :param n: number of samples to use for the calculation
    :return: numerical derivative of the given values
            Note: the first and last (n-1) samples will be NaN
    """
    x_copy = x.copy()  # use a copy of x to avoid changing the original values
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if n >= int(0.5 * len(x_copy)):
        raise ValueError("n must be less than half the length of the given values")
    if not isinstance(x_copy, pd.Series):
        # convert to pd series to use rolling window function
        x_copy = pd.Series(x_copy)
    prev_elements_sum = x_copy.rolling(n - 1).sum().shift(1)
    next_elements_sum = x_copy.rolling(n - 1).sum().shift(1 - n)
    deriv = (next_elements_sum - prev_elements_sum) / (2 * n)
    return deriv


def get_different_event_indices(is_event: np.ndarray, min_length: int = 0) -> List[np.ndarray]:
    """
    Returns a list of arrays, where each array contains the indices of a different event.
    :param is_event: np.ndarray - a boolean array, where True indicates that the sample is part of an event.
    :param min_length: int - the minimum length of an event. Events shorter than this will be ignored.

    :return: a list of arrays, where each array contains the indices of a different event.
    """
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    different_event_idxs = list(filter(lambda e: len(e) >= min_length, different_event_idxs))
    return different_event_idxs
