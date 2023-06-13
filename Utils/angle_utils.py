import numpy as np
from typing import Optional

import Utils.array_utils as au
from Utils.ScreenMonitor import ScreenMonitor


def calculate_visual_angle_velocities(x: np.ndarray, y: np.ndarray,
                                      sr: float, d: float,
                                      screen_monitor: Optional[ScreenMonitor] = None,
                                      use_radians: bool = False) -> np.ndarray:
    """
    Calculates the visual-angle velocities of the gaze data between two adjacent samples.
    :param x: 1D array of x-coordinates.
    :param y: 1D array of y-coordinates.
    :param sr: sampling rate of the data.
    :param d: distance between the monitor and the participant's eyes.
    :param screen_monitor: ScreenMonitor object.
    :param use_radians: if True, the angular velocity will be returned in radians per second.

    :return: 1D array of angular velocities (rad/s or deg/s)
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
