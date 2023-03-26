from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseFixationDetector(BaseDetector, ABC):
    """
    Baseclass for all fixation event detectors.
    """

    def __init__(self,
                 min_duration: float = conf.FIXATION_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = BaseDetector.INTER_EVENT_TIME):
        super().__init__(min_duration, sr, iet)
