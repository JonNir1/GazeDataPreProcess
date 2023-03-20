import numpy as np
from typing import Tuple

import experiment_config as conf


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


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


def pixels2deg(pixels: np.ndarray,
               distance: float = conf.SCREEN_DISTANCE,
               screen_size: Tuple[float, float] = (conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT),
               screen_resolution: Tuple[int, int] = conf.SCREEN_RESOLUTION) -> float:
    """
    Calculates the distance between two pixels in pixel-space and converts it to visual degrees.
    :param pixels: np.ndarray of shape (2, 2) - X,Y coordinates of 2 points on the screen.
    :param distance: float - the distance between the screen and the participant in cm.
    :param screen_size: (float, float) - the width and height of the screen in cm.
    :param screen_resolution: (int, int) - the resolution of the screen in pixels.

    :return: visual_angle: float - the visual angle between the two points in degrees.
    """
    x1, y1, x2, y2 = pixels.flatten()
    pixel_width_centimeters = __pixel_size_centimeters(screen_size[0], screen_resolution[0])
    horizontal_distance_centimeters = abs(x1 - x2) * pixel_width_centimeters
    pixel_height_centimeters = __pixel_size_centimeters(screen_size[1], screen_resolution[1])
    vertical_distance_centimeters = abs(y1 - y2) * pixel_height_centimeters

    visual_distance_centimeters = np.sqrt(horizontal_distance_centimeters ** 2 + vertical_distance_centimeters ** 2)
    visual_angle = __calculate_angle(distance, visual_distance_centimeters)
    return visual_angle


def pixel_width_degrees(distance: float = conf.SCREEN_DISTANCE,
                        screen_width: float = conf.SCREEN_WIDTH,
                        screen_resolution: int = conf.SCREEN_RESOLUTION[0]) -> float:
    # calculates the visual angle of the width of a pixel
    pixel_size = __pixel_size_centimeters(screen_width, screen_resolution)
    return __calculate_angle(distance, pixel_size)


def pixel_height_degrees(distance: float = conf.SCREEN_DISTANCE,
                         screen_height: float = conf.SCREEN_HEIGHT,
                         screen_resolution: int = conf.SCREEN_RESOLUTION[1]) -> float:
    # calculates the visual angle of the height of a pixel
    pixel_size = __pixel_size_centimeters(screen_height, screen_resolution)
    return __calculate_angle(distance, pixel_size)


def __pixel_size_centimeters(screen_size: float, resolution: int) -> float:
    # calculates the size of one of the pixel's edge in centimeters
    return screen_size / resolution


def __calculate_angle(distance: float, base: float) -> float:
    # calculates the visual angle of a viewer sitting at a distance from the screen and looking at a line of size base
    rad = np.arctan2(base, distance)
    return rad2deg(rad)
