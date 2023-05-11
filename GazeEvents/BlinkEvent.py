import experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):

    @property
    def is_outlier(self) -> bool:
        return self.duration < cnfg.DEFAULT_BLINK_MINIMUM_DURATION

    @classmethod
    def _event_type(cls):
        return "blink"
