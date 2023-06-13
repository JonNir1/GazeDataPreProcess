import numpy as np
import pandas as pd
from typing import Optional, Tuple

import Utils.array_utils as au
from Utils.ScreenMonitor import ScreenMonitor


def calculate_visual_angle(
        p1: Optional[Tuple[Optional[float], Optional[float]]],
        p2: Optional[Tuple[Optional[float], Optional[float]]],
        d: float,
        screen_monitor: Optional[ScreenMonitor] = None,
        use_radians=False) -> float:
    """
    Calculates the visual angle between two pixels on the screen, given that the viewer is at a distance d (in cm)
        from the screen.
    Returns the angle in degrees (or radians if `use_radians` is True).
    Returns np.nan if any of the given points is None or if any of the coordinates is None or np.nan.
    """
    if p1 is None or p2 is None:
        return np.nan
    x1, y1 = p1
    x2, y2 = p2
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return np.nan
    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
        return np.nan

    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()
    euclidean_distance = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))  # distance in pixels
    theta = np.arctan(euclidean_distance * screen_monitor.pixel_size / d)  # angle in radians
    if use_radians:
        return theta
    return np.rad2deg(theta)


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
        ang = calculate_visual_angle(p1=(x1, y1), p2=(x2, y2), d=d,
                                     screen_monitor=screen_monitor,
                                     use_radians=use_radians)
        angles.append(ang)
    angles = np.array(angles)
    return angles * sr


def calculate_pixels_from_visual_angle(d: float, angle: float, screen_monitor: Optional[ScreenMonitor] = None) -> float:
    """
    Calculates the number of pixels that correspond to a visual angle of `angle` degrees, given that the viewer is at a
        distance `d` (in cm) from the screen.

    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web Book.:
        http://www.yorku.ca/eye/visangle.htm

    :returns: the number of pixels that correspond to a visual angle of `angle` degrees.
    """
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()
    edge_cm = d * np.tan(np.deg2rad(angle / 2))  # edge size in cm
    edge_pixels = edge_cm / screen_monitor.pixel_size  # edge size in pixels
    return edge_pixels
