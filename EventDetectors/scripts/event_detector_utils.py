import numpy as np


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


