import numpy as np
from typing import Tuple, Optional

import experiment_config as cnfg


class ScreenMonitor:
    """
    Holds information about the computer screen used for the experiments.
    Default values are taken from the experiment_config.py file.
    """

    def __init__(self, width: int, height: int, refresh_rate: float, resolution: Tuple[float, float]):
        self.__width = width
        self.__height = height
        self.__refresh_rate = refresh_rate
        self.__resolution = resolution

    @classmethod
    def from_config(cls):
        return cls(cnfg.SCREEN_WIDTH, cnfg.SCREEN_HEIGHT, cnfg.SCREEN_REFRESH_RATE, cnfg.SCREEN_RESOLUTION)

    @property
    def width(self) -> int:
        # width of the screen in centimeters
        return self.__width

    @property
    def height(self) -> int:
        # height of the screen in centimeters
        return self.__height

    @property
    def refresh_rate(self) -> float:
        # refresh rate of the screen in Hz
        return self.__refresh_rate

    @property
    def resolution(self) -> Tuple[float, float]:
        # resolution of the screen, i.e. number of pixels in width and height
        return self.__resolution

    @property
    def pixel_size(self) -> float:
        # Returns the approximate size of one pixel in centimeters
        diagonal_length = np.sqrt(np.power(self.width, 2) + np.power(self.height, 2))  # size of diagonal in centimeters
        diagonal_pixels = np.sqrt(
            np.power(self.resolution[0], 2) + np.power(self.resolution[1], 2))  # size of diagonal in pixels
        return diagonal_length / diagonal_pixels

    def calc_angle_between_pixels(self, d: float,
                                  p1: Optional[Tuple[Optional[float], Optional[float]]],
                                  p2: Optional[Tuple[Optional[float], Optional[float]]],
                                  use_radian: False) -> float:
        """
        Calculates the visual angle between two pixels on the screen, given that the viewer is at a distance d (in cm)
            from the screen.
        Returns the angle in degrees (or radians if `use_radian` is True).
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

        euclidean_distance = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))  # distance in pixels
        theta = np.arctan(euclidean_distance * self.pixel_size / d)  # angle in radians
        if use_radian:
            return theta
        return np.rad2deg(theta)

    def calc_visual_angle_radius(self, d: float, angle: float) -> float:
        """
        Calculates the radius of a circle in pixels, given that the viewer is at a distance d (in cm) from the screen
            and the visual angle of the circle is angle (in degrees).
        Returns the radius in pixels.
        """
        angle_size_cm = 2 * d * np.tan(np.deg2rad(angle / 2))  # size of the angle, in cm
        angle_size_pixels = angle_size_cm / self.pixel_size  # size of the angle, in pixels
        radius = angle_size_pixels / 2  # radius of the circle, in pixels
        return radius
