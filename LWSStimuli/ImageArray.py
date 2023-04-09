import numpy as np
from enum import IntEnum, auto


class ImageArrayType(IntEnum):
    BW = auto()
    COLOR = auto()
    NOISE = auto()


class ImageArray:

    def __init__(self, array_type):
        self.__type = self.__identify_array_type(array_type)

    @staticmethod
    def __identify_array_type(array_type) -> ImageArrayType:
        if isinstance(array_type, ImageArrayType):
            return array_type
        if isinstance(array_type, str):
            if array_type.lower() == "bw":
                return ImageArrayType.BW
            elif array_type.lower() == "color":
                return ImageArrayType.COLOR
            elif array_type.lower() == "noise":
                return ImageArrayType.NOISE
        if isinstance(array_type, int):
            if array_type == 0:
                return ImageArrayType.BW
            elif array_type == 1:
                return ImageArrayType.COLOR
            elif array_type == 2:
                return ImageArrayType.NOISE
        raise ValueError("Invalid array type: {}".format(array_type))
