from abc import ABC

import experiment_config as conf
from EventDetectors.BaseDetector import BaseDetector


DEFAULT_MINIMUM_DURATION = 50


class BaseBlinkDetector(BaseDetector, ABC):
    """
    Baseclass for all blink event detectors.
    Defines these properties:
    - min_duration: minimum duration of a blink in milliseconds             (default: 50)
    - time_between_blinks: minimum time between two blinks in milliseconds  (default: 20)
    - missing_value: default value indicating missing data                  (default: experiment_config.MISSING_VALUE)
    - sampling_rate: sampling rate of the data in Hz                        (default: experiment_config.SAMPLING_RATE)
    """

    def __init__(self,
                 min_duration: float = DEFAULT_MINIMUM_DURATION,
                 missing_value: float = conf.MISSING_VALUE,
                 sr: float = conf.SAMPLING_RATE,
                 iet: float = conf.INTER_EVENT_TIME):
        super().__init__(sr, iet)
        self.__min_duration = min_duration
        self.__missing_value = missing_value

    # @abstractmethod
    # def detect(self, gaze_data: np.ndarray) -> np.ndarray:
    #     raise NotImplementedError
    # TODO: find a way to make this agnostic to function arguments

    @property
    def min_duration(self) -> float:
        return self.__min_duration

    def set_min_duration(self, min_duration: float):
        self.__min_duration = min_duration

    @property
    def missing_value(self) -> float:
        return self.__missing_value

    def set_missing_value(self, missing_value: float):
        self.__missing_value = missing_value
