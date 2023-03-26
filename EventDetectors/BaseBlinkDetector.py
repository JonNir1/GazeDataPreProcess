from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


class BaseBlinkDetector(BaseDetector, ABC):
    """
    Baseclass for all blink event detectors.
    Defines these properties:
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    - missing_value: default value indicating missing data                  (default: experiment_config.MISSING_VALUE)
    - sampling_rate: sampling rate of the data in Hz                        (default: experiment_config.SAMPLING_RATE)
    - inter_event_time: minimal time between two (same) events in ms        (default: experiment_config.INTER_EVENT_TIME)
    """

    BLINK_MINIMUM_DURATION = 50  # minimum duration of a blink in milliseconds

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments


