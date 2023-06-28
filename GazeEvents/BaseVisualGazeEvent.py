from abc import ABC
import numpy as np

from GazeEvents.BaseGazeEvent import BaseGazeEvent


class BaseVisualGazeEvent(BaseGazeEvent, ABC):
    """
    A base class for Gaze Events that contain visual information, i.e. events that require X, Y coordinates.
    For example, saccades, fixations, smooth pursuit, etc.
    """

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, viewer_distance: float):
        if not np.isfinite(viewer_distance) or viewer_distance <= 0:
            raise ValueError("viewer_distance must be a positive finite number")
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        super().__init__(timestamps=timestamps)
        self._x = x
        self._y = y
        self._viewer_distance = viewer_distance  # in cm

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not super().__eq__(other):
            return False
        if self._viewer_distance != other._viewer_distance:
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        return True

