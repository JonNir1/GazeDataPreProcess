from abc import ABC

from EventDetectors.BaseDetector import BaseDetector


class BaseSaccadeDetector(BaseDetector, ABC):
    """
    Baseclass for all saccade event detectors.
    """
