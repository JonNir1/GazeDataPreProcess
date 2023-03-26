from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseSaccadeDetector(BaseDetector, ABC):
    """
    Baseclass for all saccade event detectors.
    """

    SACCADE_MINIMUM_DURATION = 5  # minimum duration of a saccade in milliseconds

    def __init__(self,
                 sr: float,
                 min_duration: float = SACCADE_MINIMUM_DURATION,
                 iet: float = BaseDetector.INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
