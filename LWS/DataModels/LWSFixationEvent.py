import numpy as np
import pandas as pd

from GazeEvents.FixationEvent import FixationEvent


class LWSFixationEvent(FixationEvent):
    """
    A fixation event with additional information about whether it is close to any of the targets of the current LWS trial.
    """

    def __init__(self, timestamps: np.ndarray, sampling_rate: float, x: np.ndarray, y: np.ndarray):
        super().__init__(timestamps=timestamps, sampling_rate=sampling_rate, x=x, y=y)
        self.__is_close_to_targets: bool = False

    @property
    def is_close_to_targets(self) -> bool:
        return self.__is_close_to_targets

    def set_is_close_to_targets(self, is_close_to_target: bool):
        self.__is_close_to_targets = is_close_to_target

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the following index:
            - start_time: fixation's start time in milliseconds
            - end_time: fixation's end time in milliseconds
            - duration: fixation's duration in milliseconds
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - std: fixation's standard deviation (in pixels units)
            - is_close_to_targets: True if the fixation is close to any of the targets of the current trial
        """
        series = super().to_series()
        series["is_close_to_targets"] = self.is_close_to_targets
        return series
