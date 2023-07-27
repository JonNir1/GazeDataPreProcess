from enum import IntEnum


# TODO: use these instead of cnst.UNDEFINED, cnst.FIXATION, etc.
class GazeEventTypeEnum(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    BLINK = 3



