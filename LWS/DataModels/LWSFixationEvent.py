import numpy as np
import pandas as pd

from GazeEvents.FixationEvent import FixationEvent


class LWSFixationEvent(FixationEvent):
    """
    A regular FixationEvent with additional information required specifically for the LWS experiments:
        - distance_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
    """

    def __init__(self,
                 timestamps: np.ndarray, sampling_rate: float,
                 x: np.ndarray, y: np.ndarray,
                 distance_to_target: float = np.inf):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate, x=x, y=y)
        self.__distance_to_target = distance_to_target

    @property
    def distance_to_target(self) -> float:
        return self.__distance_to_target

    @distance_to_target.setter
    def distance_to_target(self, distance: float):
        self.__distance_to_target = distance

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the following index:
            - start_time: fixation's start time in milliseconds
            - end_time: fixation's end time in milliseconds
            - duration: fixation's duration in milliseconds
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - std: fixation's standard deviation (in pixels units)
            - distance_to_target: angular distance from the fixation's center of mass to the closest target's center of mass
        """
        series = super().to_series()
        series["distance_to_target"] = self.distance_to_target
        return series
