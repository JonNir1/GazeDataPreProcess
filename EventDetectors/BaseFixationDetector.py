from abc import ABC

from Config import experiment_config as cnfg
from EventDetectors.BaseDetector import BaseDetector


class BaseFixationDetector(BaseDetector, ABC):
    """
    Baseclass for all fixation event detectors.
    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - min_duration: minimum duration of a blink in milliseconds             (default: 55)
    - inter_event_time: minimal time between two (same) events in ms        (default: 5)
    """

    def __init__(self,
                 sr: float,
                 min_duration: float = cnfg.DEFAULT_FIXATION_MINIMUM_DURATION,
                 iet: float = cnfg.DEFAULT_INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
