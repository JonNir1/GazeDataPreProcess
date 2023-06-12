import numpy as np
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


def get_different_event_indices(is_event: np.ndarray) -> List[np.ndarray]:
    """
    Returns a list of arrays, where each array contains the indices of a different event.

    :param is_event: np.ndarray - a boolean array, where True indicates that the sample is part of an event.
    :return: a list of arrays, where each array contains the indices of a different event.
    """
    event_idxs = np.nonzero(is_event)[0]
    if len(event_idxs) == 0:
        return []
    event_end_idxs = np.nonzero(np.diff(event_idxs) != 1)[0]
    different_event_idxs = np.split(event_idxs, event_end_idxs + 1)  # +1 because we want to include the last index
    return different_event_idxs
