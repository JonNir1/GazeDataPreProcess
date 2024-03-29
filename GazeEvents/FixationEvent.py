import numpy as np
import pandas as pd
from typing import Tuple

import Config.experiment_config as cnfg
import Utils.angle_utils as angle_utils
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class FixationEvent(BaseVisualGazeEvent):
    _EVENT_TYPE = GazeEventTypeEnum.FIXATION
    MIN_DURATION = 55    # (milliseconds)
    MAX_DURATION = 2500  # (milliseconds)

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, pupil: np.ndarray, viewer_distance: float):
        super().__init__(timestamps=timestamps, x=x, y=y, viewer_distance=viewer_distance)
        self._pupil: np.ndarray = pupil  # pupil size (in mm)

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        # returns the mean coordinates of the fixation on the X,Y axes
        x_mean = float(np.nanmean(self._x))
        y_mean = float(np.nanmean(self._y))
        return x_mean, y_mean

    @property
    def standard_deviation(self) -> Tuple[float, float]:
        # returns the standard deviation of the fixation on the X,Y axes
        x_std = float(np.nanstd(self._x))
        y_std = float(np.nanstd(self._y))
        return x_std, y_std

    @property
    def dispersion(self) -> float:
        # returns the maximum distance between any two points in the fixation (in pixel units)
        points = np.column_stack((self._x, self._y))
        distances = np.linalg.norm(points - points[:, None], axis=-1)
        max_dist = float(np.nanmax(distances))
        return max_dist

    @property
    def mean_pupil_size(self) -> float:
        # returns the mean pupil size during the fixation (in mm)
        return float(np.nanmean(self._pupil))

    @property
    def std_pupil_size(self) -> float:
        # returns the standard deviation of the pupil size during the fixation (in mm)
        return float(np.nanstd(self._pupil))

    def get_outlier_reasons(self):
        reasons = super().get_outlier_reasons()
        # TODO: check max velocity, acceleration, dispersion
        # TODO: check if inside the screen
        return reasons

    def get_pupil_series(self, round_decimals: int = 1, zero_corrected: bool = True) -> pd.Series:
        """
        Returns a pandas Series with the event's pupil sizes (mm) and indexed by timestamps, rounded to the specified
        number of decimals. If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self.get_timestamps(round_decimals=round_decimals, zero_corrected=zero_corrected)
        return pd.Series(data=self._pupil, index=timestamps, name="pupil_size")

    def is_close_to_pixel(self, pixel: Tuple[float, float], threshold: float, threshold_units: str = 'deg') -> bool:
        """
        Returns True if the fixation's center of mass is within the given threshold away from the target pixel.
        Parameter `threshold_units` sets the type of threshold distance to measure: 'deg', 'rad', or 'px'.

        :raises ValueError: if the pixel's coordinates are not finite numbers
        :raises ValueError: if the threshold is not a finite non-negative number
        :raises ValueError: if the threshold_units is not one of 'deg', 'rad', or 'px'
        """
        if not np.isfinite(pixel[0]) or not np.isfinite(pixel[1]):
            raise ValueError(f"Pixel coordinates must be finite numbers: {pixel}")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"Threshold must be a finite non-negative number: {threshold}")
        if threshold_units not in ['deg', 'rad', 'pix']:
            raise ValueError(f"Threshold units must be 'deg'/'rad'/'px': {threshold_units}")

        center_x, center_y = self.center_of_mass
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        if threshold_units == 'px':
            distance = np.sqrt(np.power(center_x - pixel[0], 2) + np.power(center_y - pixel[1], 2))
        else:
            use_radians = threshold_units == 'rad'
            distance = angle_utils.calculate_visual_angle(p1=self.center_of_mass, p2=pixel,
                                                          d=self._viewer_distance,
                                                          pixel_size=cnfg.SCREEN_MONITOR.pixel_size,
                                                          use_radians=use_radians)
        return distance <= threshold

    def is_in_rectangle(self, top_left: Tuple[float, float], bottom_right: Tuple[float, float]) -> bool:
        """
        Returns True if the fixation's center of mass is inside the rectangle defined by the given top-left and
        bottom-right coordinates.

        :raises ValueError: if the given coordinates are not finite numbers
        """
        if not np.isfinite(top_left[0]) or not np.isfinite(top_left[1]):
            raise ValueError(f"Top-left coordinates must be finite numbers: {top_left}")
        if not np.isfinite(bottom_right[0]) or not np.isfinite(bottom_right[1]):
            raise ValueError(f"Bottom-right coordinates must be finite numbers: {bottom_right}")

        center_x, center_y = self.center_of_mass
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        if center_x < top_left[0] or center_x > bottom_right[0]:
            return False
        if center_y < top_left[1] or center_y > bottom_right[1]:
            return False
        return True

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - standard_deviation: fixation's standard deviation (in pixel units)
            - dispersion: maximum distance between any two points in the fixation (in pixels units)
            - mean_pupil_size: mean pupil size during the fixation (in mm)
            - std_pupil_size: standard deviation of the pupil size during the fixation (in mm)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["standard_deviation"] = self.standard_deviation
        series["dispersion"] = self.dispersion
        series["mean_pupil_size"] = self.mean_pupil_size
        series["std_pupil_size"] = self.std_pupil_size
        return series
