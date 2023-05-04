import os
import numpy as np
import matplotlib.pyplot as plt

from LWSStimulusBase import LWSStimulusBase
from LWSStimuli.LWSStimulusInfo import LWSStimulusInfo
from LWSStimuli.LWSStimulusTypeEnum import LWSStimulusTypeEnum


class LWSStimulus(LWSStimulusBase):
    """
    A class representing a single LWS stimulus, that is - an image array and its metadata.
    """

    def __init__(self, stim_id: int, stim_type, super_dir: str):
        super().__init__(stim_id, stim_type)
        if not os.path.isdir(super_dir):
            raise NotADirectoryError(f"Directory {super_dir} does not exist.")
        self.__super_dir = super_dir
        self.__metadata = LWSStimulusInfo.from_matlab_array(self.image_metadata_path)

    @property
    def image_array_path(self) -> str:
        return self.__path_to_file("bmp")

    @property
    def image_metadata_path(self) -> str:
        return self.__path_to_file("mat")

    def read_image_array(self) -> np.ndarray:
        return plt.imread(self.image_array_path)

    def show_image_array(self):
        color_map = 'gray' if self.stim_type == LWSStimulusTypeEnum.BW else None
        plt.imshow(self.read_image_array(), cmap=color_map)
        plt.tight_layout()
        plt.show()

    def __path_to_file(self, fmt: str) -> str:
        subdir = self.stim_type.name.lower()
        filename = f"image_{self.__stim_id}.{fmt}"
        fullpath = os.path.join(self.__super_dir, subdir, filename)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError(f"File {fullpath} does not exist.")
        return fullpath

