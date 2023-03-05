import pandas as pd
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """
    Metaclass for all detectors
    """

    @abstractmethod
    def detect(self) -> pd.DataFrame:
        raise NotImplementedError

