from enum import StrEnum


class LWSStimulusTypeEnum(StrEnum):
    BW = 'bw'
    COLOR = 'color'
    NOISE = 'noise'


class LWSSubjectSexEnum(StrEnum):
    Male = 'male'
    Female = 'female'
    Other = 'other'


class LWSSubjectDominantHandEnum(StrEnum):
    Right = 'right'
    Left = 'left'


class LWSSubjectDominantEyeEnum(StrEnum):
    Right = 'right'
    Left = 'left'
