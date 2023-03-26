import numpy as np
from typing import List

from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):

    @classmethod
    def _event_type(cls):
        return "blink"
