from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class BlinkEvent(BaseGazeEvent):
    _EVENT_TYPE = GazeEventTypeEnum.BLINK
    MIN_DURATION = 50  # minimum duration of a blink in milliseconds
    MAX_DURATION = 1000  # maximum duration of a blink in milliseconds

