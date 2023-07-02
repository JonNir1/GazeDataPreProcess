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
                 viewer_distance: float, triggers: np.ndarray, visual_angle_to_target: float = np.inf):
        super().__init__(timestamps=timestamps, x=x, y=y, viewer_distance=viewer_distance)
        triggers_with_timestamps = [(timestamps[i], triggers[i]) for i in range(len(timestamps)) if
                                    not np.isnan(triggers[i])]
        self.__triggers: List[Tuple[float, int]] = sorted(triggers_with_timestamps, key=lambda tup: tup[0])
        self.__visual_angle_to_target: float = visual_angle_to_target

    @property
    def visual_angle_to_target(self) -> float:
        return self.__visual_angle_to_target

    @visual_angle_to_target.setter
    def visual_angle_to_target(self, visual_angle: float):
        self.__visual_angle_to_target = visual_angle

    def get_triggers_with_timestamps(self) -> List[Tuple[float, int]]:
        return self.__triggers

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - trigger: list of tuples (timestamp, trigger) for each trigger that occurred during the fixation
            - visual_angle_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
        """
        series = super().to_series()
        series[cnst.TRIGGER] = self.get_triggers_with_timestamps()
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
