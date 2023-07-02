import numpy as np
from typing import Optional, Tuple

from Config import experiment_config as cnfg
import Utils.array_utils as au


def calculate_azimuth(p1: Optional[Tuple[Optional[float], Optional[float]]],
                      p2: Optional[Tuple[Optional[float], Optional[float]]],
                      zero_direction: str = 'E',
                      use_radians=False) -> float:
    """
    Calculates the counter-clockwise angle between the line starting from p1 and ending at p2, and the line starting
    from p1 and pointing in the direction of `zero_direction`.

    :param p1: the (x,y) coordinates of the starting point of the line
    :param p2: the (x,y) coordinates of the ending point of the line
    :param zero_direction: the direction of the zero angle. Must be one of 'E', 'W', 'S', 'N' (case insensitive).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the angle between the two lines, in range [0, 2*pi) or [0, 360), or np.nan if either p1 or p2 is invalid.

    :raises ValueError: if zero_direction is not one of 'E', 'W', 'S', 'N' (case insensitive).
    """

    zero_direction = zero_direction.upper()
    valid_directions = ['E', 'W', 'S', 'N']
    if zero_direction not in valid_directions:
        raise ValueError(f"zero_direction must be one of {valid_directions}")
    if not __is_valid_pixel(p1):
        return np.nan
    if not __is_valid_pixel(p2):
        return np.nan

    # calculate the angle between the line between p1 and p2, and the rightward facing x-axis
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(y1 - y2, x2 - x1)  # counter-clockwise angle line (p1, p2) and the rightward facing x-axis

    # adjust to the desired zero direction
    if zero_direction == 'W':
        angle_rad += np.pi
    elif zero_direction == 'S':
        angle_rad += np.pi / 2
    elif zero_direction == 'N':
        angle_rad -= np.pi / 2

    # make sure the angle is in range [0, 2*pi), and return
    angle_rad = angle_rad % (2 * np.pi)
    if use_radians:
        return angle_rad
    return np.rad2deg(angle_rad)


def calculate_visual_angle(
        p1: Optional[Tuple[Optional[float], Optional[float]]],
        p2: Optional[Tuple[Optional[float], Optional[float]]],
        d: float, pixel_size: float, use_radians=False) -> float:
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
    euclidean_distance = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))  # distance in pixels
    theta = np.arctan(euclidean_distance * pixel_size / d)  # angle in radians
    if use_radians:
        return theta
    return np.rad2deg(theta)


def visual_angle_to_pixels(d: float, angle: float, pixel_size: float) -> float:
    """
    Calculates the number of pixels that correspond to a visual angle of `angle` degrees, given that the viewer is at a
        distance `d` (in cm) from the screen.

    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web Book.:
        http://www.yorku.ca/eye/visangle.htm

    :returns: the number of pixels that correspond to a visual angle of `angle` degrees.
    """
    half_edge = d * np.tan(np.deg2rad(angle / 2))  # in cm
    edge_pixels = 2 * half_edge / pixel_size  # edge size in pixels
    return edge_pixels


def calculate_visual_angle_velocities(x: np.ndarray, y: np.ndarray,
                                      sr: float, d: float,
                                      use_radians: bool = False) -> np.ndarray:
    """
    Calculates the visual-angle velocities of the gaze data between two adjacent samples.
    :param x: 1D array of x-coordinates.
    :param y: 1D array of y-coordinates.
    :param sr: sampling rate of the data.
    :param d: distance between the monitor and the participant's eyes.
    :param use_radians: if True, the angular velocity will be returned in radians per second.

    :return: 1D array of angular velocities (rad/s or deg/s)
    """
    x_shifted = au.shift_array(x, 1)
    y_shifted = au.shift_array(y, 1)
    pixels = np.transpose(np.vstack([x, y, x_shifted, y_shifted]))  # shape (N, 4)

    angles = []
    for i in range(pixels.shape[0]):
        x1, y1, x2, y2 = pixels[i]
        ang = calculate_visual_angle(p1=(x1, y1), p2=(x2, y2), d=d,
                                     pixel_size=cnfg.SCREEN_MONITOR.pixel_size,
                                     use_radians=use_radians)
        angles.append(ang)
    angles = np.array(angles)
    return angles * sr


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


def calculate_visual_angle_accurate(
        P1: Optional[Tuple[Optional[float], Optional[float]]],
        P2: Optional[Tuple[Optional[float], Optional[float]]],
        d: float, use_radians=False) -> float:
    """
    UNUSED! Use calculate_visual_angle() instead.

    Calculates the visual angle between two pixels on the screen, given that the viewer is at a distance d (in cm) from
    the center of the screen, and that the (0,0) pixel is in the top-left corner of the screen.

    Returns the angle in degrees (or radians if `use_radians` is True).
    Returns np.nan if any of the given points is None or if any of the coordinates is None or np.nan.
    """

    if not __is_valid_pixel(P1):
        return np.nan
    if not __is_valid_pixel(P2):
        return np.nan

    sm = cnfg.SCREEN_MONITOR
    pixels_distance = d / sm.pixel_size  # distance from screen in pixel units
    C_x, C_y = sm.resolution[0] / 2, sm.resolution[1] / 2  # pixel coordinates of the center of the screen

    # calculate Euclidean distance between Eye and P1
    x1, y1 = P1
    CP1 = np.sqrt(np.power(x1 - C_x, 2) + np.power(y1 - C_y, 2))  # Euclidean distance between C and P1 (in pixels)
    alpha = np.arctan(CP1 / pixels_distance)  # angle between Eye and P1 (in radians)
    EP1 = CP1 / np.sin(alpha)  # Euclidean distance between Eye and P1 (in pixels)

    # calculate Euclidean distance between Eye and P2
    x2, y2 = P2
    CP2 = np.sqrt(np.power(x2 - C_x, 2) + np.power(y2 - C_y, 2))  # Euclidean distance between C and P2 (in pixels)
    beta = np.arctan(CP2 / pixels_distance)  # angle between Eye and P2 (in radians)
    EP2 = CP2 / np.sin(beta)  # Euclidean distance between Eye and P2 (in pixels)

    # calculate the visual angle between P1 and P2 (angle between EP1 and EP2) using the law of cosines
    P1P2 = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))  # Euclidean distance between P1 and P2 (in pixels)
    cos_value = (np.power(EP1, 2) + np.power(EP2, 2) - np.power(P1P2, 2)) / (2 * EP1 * EP2)
    theta = np.arccos(cos_value)  # angle between EP1 and EP2 (in radians)
    if use_radians:
        return theta
    return np.rad2deg(theta)
