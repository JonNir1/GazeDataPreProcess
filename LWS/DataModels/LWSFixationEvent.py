import numpy as np
import pandas as pd
from typing import Tuple, List

import constants as cnst
from GazeEvents.FixationEvent import FixationEvent


class LWSFixationEvent(FixationEvent):
    """
    A regular FixationEvent with additional information required specifically for the LWS experiments:
        - triggers: list of tuples (timestamp, trigger) for each trigger that occurred during the fixation
        - visual_angle_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
    """

    def __init__(self,
                 timestamps: np.ndarray, x: np.ndarray, y: np.ndarray,
                 triggers: np.ndarray, visual_angle_to_target: float = np.inf):
        super().__init__(timestamps=timestamps, x=x, y=y)
        triggers_with_timestamps = [(timestamps[i], triggers[i]) for i in range(len(timestamps)) if
                                    not np.isnan(triggers[i])]
        self.__triggers: List[Tuple[float, int]] = sorted(triggers_with_timestamps, key=lambda tup: tup[0])
        self.__visual_angle_to_target: float = visual_angle_to_target

    @property
    def triggers(self) -> List[Tuple[float, int]]:
        return self.__triggers

    @property
    def visual_angle_to_target(self) -> float:
        return self.__visual_angle_to_target

    @visual_angle_to_target.setter
    def visual_angle_to_target(self, distance: float):
        self.__visual_angle_to_target = distance

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
            - trigger: list of tuples (timestamp, trigger) for each trigger that occurred during the fixation
            - visual_angle_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
        """
        series = super().to_series()
        series[cnst.TRIGGER] = self.triggers
        series["visual_angle_to_target"] = self.visual_angle_to_target
        return series

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__triggers, other.__triggers, equal_nan=True):
            return False
        if not np.array_equal(self.__visual_angle_to_target, other.__visual_angle_to_target, equal_nan=True):
            return False
        return True
