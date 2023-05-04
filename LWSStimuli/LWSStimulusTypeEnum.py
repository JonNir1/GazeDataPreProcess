from enum import IntEnum


class LWSStimulusTypeEnum(IntEnum):
    BW = 0
    COLOR = 1
    NOISE = 2


def identify_stimulus_type(stim_type) -> LWSStimulusTypeEnum:
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
