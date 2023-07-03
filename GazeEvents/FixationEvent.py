import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent


class FixationEvent(BaseVisualGazeEvent):

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, pupil: np.ndarray, viewer_distance: float):
        super().__init__(timestamps=timestamps, x=x, y=y, viewer_distance=viewer_distance)
        self._pupil: np.ndarray = pupil  # pupil size (in mm)

    @classmethod
    def event_type(cls) -> str:
        return cnst.FIXATION

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
    def max_dispersion(self) -> float:
        # returns the maximum distance between any two points in the fixation (in pixels units)
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

    def get_pupil_series(self, round_decimals: int = 1, zero_corrected: bool = True) -> pd.Series:
        """
        Returns a pandas Series with the event's pupil sizes (mm) and indexed by timestamps, rounded to the specified
        number of decimals. If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self.get_timestamps(round_decimals=round_decimals, zero_corrected=zero_corrected)
        return pd.Series(data=self._pupil, index=timestamps, name="pupil_size")

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - standard_deviation: fixation's standard deviation (in pixel units)
            - max_dispersion: maximum distance between any two points in the fixation (in pixels units)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["standard_deviation"] = self.standard_deviation
        series["max_dispersion"] = self.max_dispersion
        series["mean_pupil_size"] = self.mean_pupil_size
        series["std_pupil_size"] = self.std_pupil_size
        return series

    @property
    def is_outlier(self) -> bool:
        if self.duration < cnfg.DEFAULT_FIXATION_MINIMUM_DURATION:
            return True
        # TODO: check max velocity
        # TODO: check max acceleration
        # TODO: check max dispersion
        return False
