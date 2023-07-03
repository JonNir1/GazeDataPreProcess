import numpy as np
import pandas as pd
from typing import List


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Returns a copy of the given array, normalized to the range [0, 1].
    """
    values_range = np.nanmax(arr) - np.nanmin(arr)
    corrected_arr = arr - np.nanmin(arr)
    return corrected_arr / values_range


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


def calculate_distribution(data: np.ndarray, nbins: int, min_threshold: float = 0) -> (np.ndarray, np.ndarray):
    """
    Calculates the distribution of the given data, and returns the percentages and centers of the bins. Ignores bins
    with percentage less than min_threshold.
    """
    counts, edges = np.histogram(data, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    percentages = 100 * counts / np.sum(counts)
    assert len(percentages) == len(centers), "Percentages and centers must have same length"

    centers = centers[percentages >= min_threshold]
    percentages = percentages[percentages >= min_threshold]
    return percentages, centers


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


def median_standard_deviation(x: np.ndarray, min_sd: float = 1e-6) -> float:
    """
    Calculates the median-based standard deviation of the given values.
    :param x: values to calculate the median standard deviation for
    :param min_sd: minimum standard deviation to return
    :return: median standard deviation
    """
    assert min_sd > 0, "min_sd must be greater than 0"
    squared_median = np.power(np.nanmedian(x), 2)
    median_of_squared = np.nanmedian(np.power(x, 2))
    sd = np.sqrt(median_of_squared - squared_median)
    return max(sd, min_sd)


def get_chunk_indices(bool_arr: np.ndarray, min_length: int = 0) -> List[np.ndarray]:
    """
    Returns a list of arrays, where each array contains the indices of a different "chunk", i.e. a sequence of True values.
    :param bool_arr: np.ndarray - a boolean array, where True indicates that the sample is part of a chunk.
    :param min_length: int - the minimum length of a chunk. Chunks shorter than this will be ignored.

    :return: a list of arrays, where each array contains the indices of a different chunk.
    """
    chunk_idxs = np.nonzero(bool_arr)[0]
    if len(chunk_idxs) == 0:
        return []
    chunk_end_idxs = np.nonzero(np.diff(chunk_idxs) != 1)[0]
    different_chunk_idxs = np.split(chunk_idxs, chunk_end_idxs + 1)  # +1 because we want to include the last index
    different_chunk_idxs = list(filter(lambda e: len(e) >= min_length, different_chunk_idxs))
    return different_chunk_idxs


def distance_between_subsequent_pixels(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between subsequent pixels in the given x and y coordinates.
    :param x: x coordinates
    :param y: y coordinates
    :return: distance between subsequent pixels
    """
    assert len(x) == len(y), "x and y must be of the same length"
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    dist = np.sqrt(np.power(x_diff, 2) + np.power(y_diff, 2))
    return dist
