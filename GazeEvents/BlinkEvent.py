import numpy as np
from typing import List

from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):

    @staticmethod
    def extract_blink_events(timestamps: np.ndarray, is_fixation: np.ndarray,
                             sampling_rate: float) -> List["BlinkEvent"]:
        """
        Extracts fixation events from the given data and returns a list of FixationEvent objects.
        """
        if len(timestamps) != len(is_fixation):
            raise ValueError("Arrays of timestamps, x, y and is_fixation must have the same length")
        different_event_idxs = BaseEvent._split_samples_between_events(is_fixation)
        blink_events = [BlinkEvent(timestamps=timestamps[idxs],
                                   sampling_rate=sampling_rate) for idxs in different_event_idxs]
        return blink_events

    @classmethod
    def _event_type(cls):
        return "blink"
