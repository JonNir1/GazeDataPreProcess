import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from LWS_DataModels.LWSStimuli.LWSStimulusBase import LWSStimulusBase


class LWSArrayStimulusInfo(LWSStimulusBase):
    """
    This class represents the metadata about a LWS stimulus:
        - pixel location of image centers
        - full paths to image files that comprise the stimulus (image array)
        - categories of images in the stimulus (faces, animals, etc.)
        - whether each image is a target image
    """

    def __init__(self, stim_id: int, stim_type,
                 image_paths: np.ndarray, image_centers: np.ndarray,
                 image_categories: np.ndarray, is_target_image: np.ndarray):
        super().__init__(stim_id, stim_type)
        self.__image_centers = image_centers
        self.__image_paths = image_paths
        self.__image_categories = image_categories
        self.__is_target_image = is_target_image

    @staticmethod
    def from_matlab_array(file_path: str) -> "LWSArrayStimulusInfo":
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.endswith(".mat"):
            raise ValueError(f"File {file_path} is not a .mat file.")

        f = loadmat(file_path)
        stim_id = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        stim_type_str = os.path.basename(os.path.dirname(file_path))
        stim_type = LWSArrayStimulusInfo._identify_stimulus_type(stim_type_str)

        mat = f["imageInfo"]
        image_paths = np.vectorize(lambda arr: arr[0])(mat["stimInArray"][0][0])  # shape (r, c)
        image_centers = np.stack(np.vectorize(lambda arr: tuple(arr[0]))(mat["stimCenters"][0][0]), axis=2)  # shape (r, c, 2)
        image_categories = mat["categoryInArray"][0][0].astype(int)  # shape (r, c)
        is_target_image = mat["targetsInArray"][0][0].astype(bool)    # shape (r, c)

        return LWSArrayStimulusInfo(stim_id=stim_id, stim_type=stim_type,
                                    image_paths=image_paths, image_centers=image_centers,
                                    image_categories=image_categories, is_target_image=is_target_image)

    @property
    def num_targets(self) -> int:
        return int(np.sum(self.__is_target_image))

    def get_target_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the following columns:
            - image_path: full path to the image file
            - image_category: category of the image
            - center_x: x coordinate of the image center
            - center_y: y coordinate of the image center
        """
        target_indices = np.where(self.__is_target_image)
        target_data = pd.DataFrame({
            "image_path": self.__image_paths[target_indices],
            "image_category": self.__image_categories[target_indices],
            "center_x": self.__image_centers[target_indices][:, 0],
            "center_y": self.__image_centers[target_indices][:, 1]
        })
        return target_data
