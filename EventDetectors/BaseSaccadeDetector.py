from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseSaccadeDetector(BaseDetector, ABC):
    """
    Baseclass for all saccade event detectors.
    """

    def __init__(self,
                 min_duration: float = conf.SACCADE_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = conf.INTER_EVENT_TIME):
        super().__init__(min_duration, sr, iet)
