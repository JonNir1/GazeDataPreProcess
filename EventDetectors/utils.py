from math import ceil
import numpy as np
import statistics as st

import constants as c


def calculate_sampling_rate(timestamps: np.ndarray) -> float:
    """
    Calculates the sampling rate of the given timestamps.
    :param timestamps: 1D array of timestamps in microseconds
    :return: sampling rate in Hz
    """
    return c.MICROSECONDS / st.mode(np.diff(timestamps))


def calculate_minimum_sample_count(min_duration: float, sampling_rate: float) -> int:
    """
    Calculates the minimum number of consecutive samples that must be present to be considered an event.
    :param min_duration: minimum duration of an event in milliseconds
    :param sampling_rate: sampling rate of the data in Hz
    :return: minimum number of consecutive samples
    """
    return ceil(min_duration * sampling_rate / c.MILLISECONDS_PER_SECOND)
