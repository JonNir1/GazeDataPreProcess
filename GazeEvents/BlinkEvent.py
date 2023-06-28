import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class BlinkEvent(BaseGazeEvent):

    @property
    def is_outlier(self) -> bool:
        return self.duration < cnfg.DEFAULT_BLINK_MINIMUM_DURATION

    @classmethod
    def event_type(cls):
        return cnst.BLINK
