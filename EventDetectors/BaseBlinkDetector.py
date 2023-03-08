from abc import ABC

from EventDetectors.BaseDetector import BaseDetector


class BaseBlinkDetector(BaseDetector, ABC):
    """
    Baseclass for all blink event detectors.
    Defines these properties:
    - sampling_rate: sampling rate of the data in Hz                        (default: experiment_config.SAMPLING_RATE)
    - missing_value: default value indicating missing data                  (default: experiment_config.MISSING_VALUE)
    - time_between_blinks: minimum time between two blinks in milliseconds  (default: 20)
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    """

    def __init__(self, time_between_blinks: float = 20, min_duration: float = 50):
        super().__init__()
        self.__time_between_blinks = time_between_blinks
        self.__min_duration = min_duration

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments

    @property
    def time_between_blinks(self) -> float:
        return self.__time_between_blinks

    @property
    def min_duration(self) -> float:
        return self.__min_duration
