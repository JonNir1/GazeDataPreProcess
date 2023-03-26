from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector

DEFAULT_BLINK_MINIMUM_DURATION = 50  # minimum duration of a blink in milliseconds


class BaseBlinkDetector(BaseDetector, ABC):
    """
    Baseclass for all blink event detectors.
    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    - inter_event_time: minimal time between two (same) events in ms        (default: 5)
    """

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments


