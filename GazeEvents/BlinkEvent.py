from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):

    @classmethod
    def _event_type(cls):
        return "blink"
