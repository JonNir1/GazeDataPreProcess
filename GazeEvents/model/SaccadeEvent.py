import numpy as np

from GazeEvents.model.BaseEvent import BaseEvent


class SaccadeEvent(BaseEvent):

    @classmethod
    def _event_type(cls):
        return "saccade"

