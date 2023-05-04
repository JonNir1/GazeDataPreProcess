from enum import IntEnum


class LWSStimulusType(IntEnum):
    BW = 0
    COLOR = 1
    NOISE = 2


class LWSStimulus:

    def __init__(self, stim_id: int, stim_type):
        self.__stim_id = stim_id
        self.__stim_type = self.__identify_stimulus_type(stim_type)

    @property
    def stim_id(self) -> int:
        return self.__stim_id

    @property
    def stim_type(self) -> LWSStimulusType:
        return self.__stim_type

    @staticmethod
    def __identify_stimulus_type(stim_type) -> LWSStimulusType:
        if isinstance(stim_type, LWSStimulusType):
            return stim_type
        if stim_type == "BW" or stim_type == "bw" or stim_type == 0:
            return LWSStimulusType.BW
        if stim_type == "COLOR" or stim_type == "color" or stim_type == 1:
            return LWSStimulusType.COLOR
        if stim_type == "NOISE" or stim_type == "noise" or stim_type == 2:
            return LWSStimulusType.NOISE
        raise ValueError(f"Stimulus type {stim_type} is not valid.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

