from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseFixationDetector(BaseDetector, ABC):
    """
    Baseclass for all fixation event detectors.
    """

    FIXATION_MINIMUM_DURATION = 55  # minimum duration of a fixation in milliseconds

    def __init__(self,
                 min_duration: float = FIXATION_MINIMUM_DURATION,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = BaseDetector.INTER_EVENT_TIME):
        super().__init__(min_duration, sr, iet)
