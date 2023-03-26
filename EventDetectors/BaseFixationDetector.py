from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseFixationDetector(BaseDetector, ABC):
    """
    Baseclass for all fixation event detectors.
    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - min_duration: minimum duration of a blink in milliseconds             (default: 55)
    - inter_event_time: minimal time between two (same) events in ms        (default: 5)
    """

    DEFAULT_FIXATION_MINIMUM_DURATION = 55  # minimum duration of a fixation in milliseconds

    def __init__(self,
                 sr: float,
                 min_duration: float = DEFAULT_FIXATION_MINIMUM_DURATION,
                 iet: float = BaseDetector.DEFAULT_INTER_EVENT_TIME):
        super().__init__(sr=sr, min_duration=min_duration, iet=iet)
