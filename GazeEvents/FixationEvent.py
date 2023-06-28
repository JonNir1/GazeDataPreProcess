import numpy as np
import pandas as pd
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseVisualGazeEvent import BaseVisualGazeEvent


class FixationEvent(BaseVisualGazeEvent):

    @classmethod
    def event_type(cls) -> str:
        return cnst.FIXATION

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        # returns the mean coordinates of the fixation on the X,Y axes
        return np.nanmean(self.__x), np.nanmean(self.__y)

    @property
    def std(self) -> Tuple[float, float]:
        # returns the standard deviation of the fixation on the X,Y axes
        return np.nanstd(self.__x), np.nanstd(self.__y)

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - std: fixation's standard deviation (in pixels units)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["std"] = self.std
        return series

    @property
    def is_outlier(self) -> bool:
        if self.duration < cnfg.DEFAULT_FIXATION_MINIMUM_DURATION:
            return True
        # TODO: check max velocity
        # TODO: check max acceleration
        # TODO: check max dispersion
        return False
