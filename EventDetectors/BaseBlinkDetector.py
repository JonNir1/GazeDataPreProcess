from abc import ABC

from EventDetectors.BaseDetector import BaseDetector


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


