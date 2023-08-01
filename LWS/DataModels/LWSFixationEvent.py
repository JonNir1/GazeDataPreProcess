import numpy as np
import pandas as pd
from typing import Tuple, List

import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.FixationEvent import FixationEvent


class LWSFixationEvent(FixationEvent):
    """
    A regular FixationEvent with additional information required specifically for the LWS experiments:
        - triggers: list of tuples (timestamp, trigger) for each trigger that occurred during the fixation
        - visual_angle_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
    """

    def __init__(self,
                 timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, pupil: np.ndarray,
                 viewer_distance: float, triggers: np.ndarray, visual_angle_to_targets: List[float] = None):
        super().__init__(timestamps=timestamps, x=x, y=y, pupil=pupil, viewer_distance=viewer_distance)
        triggers_with_timestamps = [(timestamps[i], triggers[i]) for i in range(len(timestamps)) if
                                    not np.isnan(triggers[i])]
        self.__triggers: List[Tuple[float, int]] = sorted(triggers_with_timestamps, key=lambda tup: tup[0])
        self.__visual_angle_to_targets: List[float] = [] if visual_angle_to_targets is None else visual_angle_to_targets

    @property
    def is_mark_target_attempt(self) -> bool:
        """
        Returns true if the subject attempted to mark a target during the fixation.
        """
        mark_target_triggers = self.get_triggers_with_timestamps(
            values=[cnfg.MARK_TARGET_SUCCESSFUL_TRIGGER, cnfg.MARK_TARGET_UNSUCCESSFUL_TRIGGER]
        )
        return len(mark_target_triggers) > 0

    @property
    def visual_angle_to_targets(self) -> List[float]:
        return self.__visual_angle_to_targets

    @visual_angle_to_targets.setter
    def visual_angle_to_targets(self, visual_angles: List[float]):
        self.__visual_angle_to_targets = visual_angles

    @property
    def visual_angle_to_closest_target(self) -> float:
        min_dist = np.nanmin(self.__visual_angle_to_targets)
        if np.isfinite(min_dist):
            return float(min_dist)
        return np.nan

    def get_triggers_with_timestamps(self, values: List[int] = None) -> List[Tuple[float, int]]:
        """
        Returns a list of tuples (timestamp, trigger) for each trigger that occurred during the fixation.
        If `values` is not None, returns only triggers whose value is in `values`.
        """
        if values is None:
            return self.__triggers
        return [tup for tup in self.__triggers if tup[1] in values]

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - trigger: list of tuples (timestamp, trigger) for each trigger that occurred during the fixation
            - visual_angle_to_targets: angular distance from the fixation's center of mass to each target's center of mass
        """
        series = super().to_series()
        series[cnst.TRIGGER] = self.get_triggers_with_timestamps()
        series["visual_angle_to_targets"] = self.visual_angle_to_targets
        return series

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.__triggers, other.__triggers, equal_nan=True):
            return False
        if not np.array_equal(self.visual_angle_to_targets, other.visual_angle_to_targets, equal_nan=True):
            return False
        return True
