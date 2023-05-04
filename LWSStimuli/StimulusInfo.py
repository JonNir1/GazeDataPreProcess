import os
import numpy as np
from enum import IntEnum
from scipy.io import loadmat


class ImageArrayType(IntEnum):
    BW = 0
    COLOR = 1
    NOISE = 2


class StimulusInfo:

    def __init__(self, array_type, array_id: int,
                 image_paths: np.ndarray, image_centers: np.ndarray,
                 image_categories: np.ndarray, is_target_image: np.ndarray):
        self.__type = self.__identify_array_type(array_type)
        self.__id = array_id
        self.__image_centers = image_centers
        self.__image_paths = image_paths
        self.__image_categories = image_categories
        self.__is_target_image = is_target_image

    @staticmethod
    def from_file(file_path: str) -> "StimulusInfo":
        f = loadmat(file_path)
        array_type_str = os.path.basename(os.path.dirname(file_path))
        array_type = StimulusInfo.__identify_array_type(array_type_str)
        array_id = int(os.path.basename(file_path).split('_')[1].split('.')[0])

        mat = f["imageInfo"]
        image_paths = np.vectorize(lambda arr: arr[0])(mat["stimInArray"][0][0])  # shape (r, c)
        image_centers = np.stack(np.vectorize(lambda arr: tuple(arr[0]))(mat["stimCenters"][0][0]), axis=2)  # shape (r, c, 2)
        image_categories = mat["categoryInArray"][0][0].astype(int)  # shape (r, c)
        is_target_image = mat["targetsInArray"][0][0].astype(bool)    # shape (r, c)

        return StimulusInfo(array_type=array_type, array_id=array_id,
                            image_paths=image_paths, image_centers=image_centers,
                            image_categories=image_categories, is_target_image=is_target_image)

    @property
    def array_id(self) -> str:
        return f"{self.__type.name.upper()}{self.__id}"

    @property
    def num_targets(self) -> int:
        return np.sum(self.__is_target_image)

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

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.array_id}"
