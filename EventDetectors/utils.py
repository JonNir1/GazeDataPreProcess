from math import ceil

import constants as c


def calculate_minimum_sample_count(min_duration: float, sampling_rate: float) -> int:
    """
    Calculates the minimum number of consecutive samples that must be present to be considered an event.
    :param min_duration: minimum duration of an event in milliseconds
    :param sampling_rate: sampling rate of the data in Hz
    :return: minimum number of consecutive samples
    """
    return ceil(min_duration * sampling_rate / c.MILLISECONDS_PER_SECOND)
