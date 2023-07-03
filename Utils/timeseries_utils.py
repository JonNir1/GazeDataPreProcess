import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List, Tuple

import Utils.array_utils as au


def merge_timeseries(all_series: List[pd.Series], time_decimals: int = 1) -> pd.DataFrame:
    """
    Merges the given timeseries into a single dataframe, resampling them to the same timestamps.
    :param all_series: a list of pd.Series objects, each representing a timeseries, indexed by timestamps (floats, ms) and
        containing a single measured value (float).
    :param time_decimals: the number of decimals to round the timestamps to.

    :return: a pd.DataFrame containing the merged timeseries, indexed by timestamps (floats, ms) and with each column
        containing measurements from a single timeseries.
    """
    new_series = []
    for i, s in enumerate(all_series):
        rounded_index = np.round(s.index, decimals=time_decimals)
        resampled = s.reindex(rounded_index, method="nearest")
        resampled.name = i
        new_series.append(resampled)
    df = pd.concat(new_series, axis=1).sort_index()
    return df


def interpolate_and_merge_timeseries(all_series: List[pd.Series], interpolation_kind: str = 'linear') -> pd.DataFrame:
    """
    Interpolates the given timeseries to the same number of samples and merges them into a single dataframe.
    Each element of all_series is a pd.Series object, indexed by timestamps (floats, ms) and containing a single
    measured value (float). The timestamps of the returned dataframe are normalized to [0,1] and the values are
    interpolated to the same number of samples.
    """
    max_length = max([len(s) for s in all_series])
    new_series = []
    for i, s in enumerate(all_series):
        interpolated_time, interpolated_values = interpolate_samples(s.index.values, s.values, max_length,
                                                                     interpolation_kind=interpolation_kind)
        interpolated_series = pd.Series(data=interpolated_values, index=interpolated_time)
        interpolated_series.name = i
        new_series.append(interpolated_series)
    df = pd.concat(new_series, axis=1).sort_index()
    df.dropna(inplace=True, how='all')  # drop rows with all NaN values
    df.index = np.round(df.index * 100, decimals=1)  # change index to range [0,100] to represent percentage of time
    return df


def interpolate_samples(t: np.ndarray, samples: np.ndarray, num_samples: int,
                        interpolation_kind: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes the timestamps to range [0,1] and interpolates the values to the given number of samples.

    :param t: timestamps of the timeseries
    :param samples: sampled values of the timeseries
    :param num_samples: number of samples to interpolate to
    :param interpolation_kind: the kind of interpolation to use. See scipy.interpolate.interp1d for more details.

    :return: interpolated_time: np.ndarray - the interpolated timestamps
             interpolated_values: np.ndarray - the interpolated values

    :raises ValueError if the number of timestamps and samples are not equal.
    """
    if len(t) != len(samples):
        raise ValueError("The number of timestamps and samples must be equal.")
    normalized_timestamps = au.normalize_array(t)  # normalize to [0, 1]
    interpolator = interp1d(normalized_timestamps, samples, kind=interpolation_kind)
    interpolated_time = np.linspace(0, 1, num_samples)
    interpolated_values = interpolator(interpolated_time)
    return interpolated_time, interpolated_values

