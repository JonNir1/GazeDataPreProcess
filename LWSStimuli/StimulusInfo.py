import os
import numpy as np
from scipy.io import loadmat

from LWSStimuli.LWSStimulusTypeEnum import LWSStimulusTypeEnum, identify_stimulus_type


class StimulusInfo:

    def __init__(self, stim_id: int, stim_type,
                 image_paths: np.ndarray, image_centers: np.ndarray,
                 image_categories: np.ndarray, is_target_image: np.ndarray):
        self.__stim_id = stim_id
        self.__stim_type = identify_stimulus_type(stim_type)
        self.__image_centers = image_centers
        self.__image_paths = image_paths
        self.__image_categories = image_categories
        self.__is_target_image = is_target_image

    @staticmethod
    def from_matlab_array(file_path: str) -> "StimulusInfo":
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.endswith(".mat"):
            raise ValueError(f"File {file_path} is not a .mat file.")

        f = loadmat(file_path)
        array_type_str = os.path.basename(os.path.dirname(file_path))
        array_type = identify_stimulus_type(array_type_str)
        array_id = int(os.path.basename(file_path).split('_')[1].split('.')[0])

        mat = f["imageInfo"]
        image_paths = np.vectorize(lambda arr: arr[0])(mat["stimInArray"][0][0])  # shape (r, c)
        image_centers = np.stack(np.vectorize(lambda arr: tuple(arr[0]))(mat["stimCenters"][0][0]), axis=2)  # shape (r, c, 2)
        image_categories = mat["categoryInArray"][0][0].astype(int)  # shape (r, c)
        is_target_image = mat["targetsInArray"][0][0].astype(bool)    # shape (r, c)

        return StimulusInfo(stim_id=array_id, stim_type=array_type,
                            image_paths=image_paths, image_centers=image_centers,
                            image_categories=image_categories, is_target_image=is_target_image)

    @property
    def stim_id(self) -> int:
        return self.__stim_id

    @property
    def stim_type(self) -> LWSStimulusTypeEnum:
        return self.__stim_type

    @property
    def num_targets(self) -> int:
        return int(np.sum(self.__is_target_image))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"
