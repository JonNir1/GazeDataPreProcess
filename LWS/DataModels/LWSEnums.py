from enum import StrEnum


class LWSStimulusTypeEnum(StrEnum):
    BW = 'bw'
    COLOR = 'color'
    NOISE = 'noise'


class LWSSubjectSexEnum(StrEnum):
    Male = 'Male'
    Female = 'Female'
    Other = 'Other'


class LWSSubjectDominantHandEnum(StrEnum):
    Right = 'Right'
    Left = 'Left'


class LWSSubjectDominantEyeEnum(StrEnum):
    Right = 'Right'
    Left = 'Left'