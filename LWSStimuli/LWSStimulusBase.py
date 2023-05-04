import os
from abc import ABC

from LWSStimuli.LWSStimulusTypeEnum import LWSStimulusTypeEnum


class LWSStimulusBase(ABC):
    """
    Abstract base class for all classes that hold information about LWS stimuli
    """

    def __init__(self, stim_id: int, stim_type):
        self.__stim_id = stim_id
        self.__stim_type = LWSStimulusBase._identify_stimulus_type(stim_type)

    @property
    def stim_id(self) -> int:
        return self.__stim_id

    @property
    def stim_type(self) -> LWSStimulusTypeEnum:
        return self.__stim_type

    @staticmethod
    def _identify_stimulus_type(stim_type) -> LWSStimulusTypeEnum:
        """
        Returns the correct value of LWSStimulusTypeEnum based on the input.
        :raises ValueError: if stim_type is not a valid value.
        """
        if isinstance(stim_type, LWSStimulusTypeEnum):
            return stim_type
        if stim_type == "BW" or stim_type == "bw" or stim_type == 0:
            return LWSStimulusTypeEnum.BW
        if stim_type == "COLOR" or stim_type == "color" or stim_type == 1:
            return LWSStimulusTypeEnum.COLOR
        if stim_type == "NOISE" or stim_type == "noise" or stim_type == 2:
            return LWSStimulusTypeEnum.NOISE
        raise ValueError(f"Stimulus type {stim_type} is not valid.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

