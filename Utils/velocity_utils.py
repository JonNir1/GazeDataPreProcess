import numpy as np
import pandas as pd
from typing import Optional

import Utils.array_utils as au
from Utils.ScreenMonitor import ScreenMonitor


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


def calculate_angular_velocity(x: np.ndarray, y: np.ndarray,
                               sr: float, d: float,
                               screen_monitor: Optional[ScreenMonitor] = None,
                               use_radians: bool = False) -> np.ndarray:
    """
    Calculates the angular velocity of the gaze data between two adjacent samples.
    :param x: 1D array of x-coordinates.
    :param y: 1D array of y-coordinates.
    :param sr: sampling rate of the data.
    :param d: distance between the monitor and the participant's eyes.
    :param screen_monitor: ScreenMonitor object.
    :param use_radians: if True, the angular velocity will be returned in radians per second.

    :return: 1D array of angular velocities.
    """
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()
    x_shifted = au.shift_array(x, 1)
    y_shifted = au.shift_array(y, 1)
    pixels = np.transpose(np.vstack([x, y, x_shifted, y_shifted]))  # shape (N, 4)

    angles = []
    for i in range(pixels.shape[0]):
        x1, y1, x2, y2 = pixels[i]
        ang = screen_monitor.calc_angle_between_pixels(d=d, p1=(x1, x2), p2=(y1, y2), use_radians=use_radians)
        angles.append(ang)
    angles = np.array(angles)
    return angles * sr
