import os
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt


class LWSStimulusType(IntEnum):
    BW = 0
    COLOR = 1
    NOISE = 2


class LWSStimulus:

    def __init__(self, stim_id: int, stim_type, directory: str):
        self.__stim_id = stim_id
        self.__stim_type = self.__identify_stimulus_type(stim_type)
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory {directory} does not exist.")
        self.__stimulus_directory = directory

    @property
    def stim_id(self) -> int:
        return self.__stim_id

    @property
    def stim_type(self) -> LWSStimulusType:
        return self.__stim_type

    @property
    def image_array_path(self) -> str:
        return self.__path_to_file("bmp")

    @property
    def image_info_path(self) -> str:
        return self.__path_to_file("mat")

    def read_image_array(self) -> np.ndarray:
        return plt.imread(self.image_array_path)

    def show_image_array(self):
        color_map = 'gray' if self.stim_type == LWSStimulusType.BW else None
        plt.imshow(self.read_image_array(), cmap=color_map)
        plt.tight_layout()
        plt.show()

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

    def __path_to_file(self, format: str) -> str:
        subdir = self.stim_type.name.lower()
        filename = f"image_{self.__stim_id}.{format}"
        fullpath = os.path.join(self.__stimulus_directory, subdir, filename)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError(f"File {fullpath} does not exist.")
        return fullpath

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

