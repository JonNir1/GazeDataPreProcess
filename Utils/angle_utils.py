import numpy as np
from typing import Optional, Tuple

import Utils.array_utils as au
from Utils.ScreenMonitor import ScreenMonitor


def calculate_azimuth(p1: Optional[Tuple[Optional[float], Optional[float]]],
                      p2: Optional[Tuple[Optional[float], Optional[float]]],
                      use_radians=False) -> float:
    # TODO: fix implementation so that it takes into account the different sizes of the pixels in the x and y axes
    """
    Calculates the counter-clockwise angle between the line starting from (0,0) and ending at p1, and the line starting
    from (0,0) and ending at p2.
    The axes are defined by pixel coordinates, with the origin (0,0) in the top-left corner of the screen,
        the positive x-axis pointing right, and the positive y-axis pointing down.
    Angles are in range [0, 2*pi) or [0, 360).

    Returns the angle in degrees (or radians if `use_radians` is True).
    Returns np.nan if any of the given points is None or if any of the coordinates is None or np.nan.
    """
    if not __is_valid_pixel(p1):
        return np.nan
    if not __is_valid_pixel(p2):
        return np.nan
    x1, y1 = p1
    x2, y2 = p2
    rad1 = np.arctan2(y1, x1)  # clockwise angle from the positive x-axis
    rad2 = np.arctan2(y2, x2)  # clockwise angle from the positive x-axis
    diff = rad1 - rad2         # counter-clockwise angle from p1 to p2
    diff = diff % (2 * np.pi)  # make sure the angle is in range [0, 2*pi)
    if use_radians:
        return diff
    return np.rad2deg(diff)


def calculate_visual_angle(
        p1: Optional[Tuple[Optional[float], Optional[float]]],
        p2: Optional[Tuple[Optional[float], Optional[float]]],
        d: float,
        screen_monitor: Optional[ScreenMonitor] = None,
        use_radians=False) -> float:
    # TODO: fix implementation so that:
    #  1. it assumes y-axis is pointing **down** (and not up) and that x-axis is pointing **right**
    #  2. ii assumes the distance d is from the **center** of the screen to the participant's eyes
    #  3. it takes into account the different sizes of the pixels in the x and y axes
    """
    Calculates the visual angle between two pixels on the screen, given that the viewer is at a distance d (in cm)
        from the screen.
    Returns the angle in degrees (or radians if `use_radians` is True).
    Returns np.nan if any of the given points is None or if any of the coordinates is None or np.nan.
    """
    if not __is_valid_pixel(p1):
        return np.nan
    if not __is_valid_pixel(p2):
        return np.nan

    x1, y1 = p1
    x2, y2 = p2
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


def visual_angle_to_pixels(d: float, angle: float, screen_monitor: Optional[ScreenMonitor] = None) -> float:
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


def __is_valid_pixel(p: Optional[Tuple[Optional[float], Optional[float]]]) -> bool:
    # Returns True if the given pixel is valid (i.e. not None and not np.nan).
    if p is None:
        return False
    x, y = p
    if x is None or y is None:
        return False
    if np.isnan(x) or np.isnan(y):
        return False
    return True
