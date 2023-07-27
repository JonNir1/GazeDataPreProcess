import constants as cnst
from Config import experiment_config as cnfg
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class BlinkEvent(BaseGazeEvent):
    MIN_DURATION = 50  # minimum duration of a blink in milliseconds
    MAX_DURATION = 1000  # maximum duration of a blink in milliseconds

    @classmethod
    def event_type(cls):
        return cnst.BLINK
