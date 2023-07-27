import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class BlinkEvent(BaseGazeEvent):

    @classmethod
    def event_type(cls):
        return cnst.BLINK

    def get_outlier_reasons(self):
        reasons = []
        if self.duration < cnfg.DEFAULT_BLINK_MINIMUM_DURATION:
            reasons.append(cnst.DURATION)
        return reasons
