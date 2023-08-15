from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class BlinkEvent(BaseGazeEvent):
    _EVENT_TYPE = GazeEventTypeEnum.BLINK
    MIN_DURATION = 50    # (milliseconds)
    MAX_DURATION = 500  # (milliseconds)

