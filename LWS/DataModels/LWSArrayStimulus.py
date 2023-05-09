import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

import experiment_config as cnfg
from LWS.DataModels.LWSEnums import LWSStimulusTypeEnum


class LWSArrayStimulus:
    """
    This class represents a single LWS icon-array stimulus:
        - the stimulus image (as np.ndarray)
        - pixel location of icon centers
        - full paths to icon files that comprise the stimulus (image array)
        - categories of icon in the stimulus (faces, animals, etc.)
        - whether each icon is a target image
    """
    # TODO: encode as hdf5 file

    def __init__(self, stim_id: int, stim_type, image: np.ndarray,
                 icon_paths: np.ndarray, icon_centers: np.ndarray,
                 icon_categories: np.ndarray, is_target_icon: np.ndarray):
        self.__stim_id = stim_id
        self.__stim_type = self.__identify_stimulus_type(stim_type)
        self.__image = image
        self.__icon_paths = icon_paths
        self.__icon_centers = icon_centers
        self.__icon_categories = icon_categories
        self.__is_target_icon = is_target_icon

    @staticmethod
    def from_paths(image_path: str, metadata_path: str) -> "LWSArrayStimulus":
        """
        Reads the stimulus from the given paths and returns a LWSArrayStimulus object.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File {image_path} does not exist.")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"File {metadata_path} does not exist.")

        stim_id = int(os.path.basename(image_path).split('_')[1].split('.')[0])
        stim_type_str = os.path.basename(os.path.dirname(image_path))
        stim_type = LWSArrayStimulus.__identify_stimulus_type(stim_type_str)
        image = plt.imread(image_path)

        f = loadmat(metadata_path)
        mat = f["imageInfo"]
        icon_paths = np.vectorize(lambda arr: arr[0])(mat["stimInArray"][0][0])  # shape (r, c)
        icon_centers = np.stack(np.vectorize(lambda arr: tuple(arr[0]))(mat["stimCenters"][0][0]), axis=2)  # shape (r, c, 2)
        icon_categories = mat["categoryInArray"][0][0].astype(int)  # shape (r, c)
        is_target_icon = mat["targetsInArray"][0][0].astype(bool)  # shape (r, c)

        return LWSArrayStimulus(stim_id, stim_type, image, icon_paths, icon_centers, icon_categories, is_target_icon)

    @staticmethod
    def from_stimulus_name(stim_id: int, stim_type: str, stim_directory: str = cnfg.STIMULI_DIR) -> "LWSArrayStimulus":
        """
        Reads the stimulus based on the provided stimulus ID and type and returns a LWSArrayStimulus object.
        """
        stim_type = LWSArrayStimulus.__identify_stimulus_type(stim_type)
        image_path = os.path.join(stim_directory, stim_type, f"image_{stim_id}.bmp")
        metadata_path = os.path.join(stim_directory, stim_type, f"image_{stim_id}.mat")
        return LWSArrayStimulus.from_paths(image_path, metadata_path)

    @property
    def stim_id(self) -> int:
        return self.__stim_id

    @property
    def stim_type(self) -> LWSStimulusTypeEnum:
        return self.__stim_type

    @property
    def num_targets(self) -> int:
        return int(np.sum(self.__is_target_icon))

    def show(self):
        # TODO: add circles around target icons
        color_map = 'gray' if self.stim_type == LWSStimulusTypeEnum.BW else None
        plt.imshow(self.__image, cmap=color_map)
        plt.tight_layout()
        plt.show()

    def get_target_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the following columns:
            - icon_path: full path to the icon file
            - icon_category: category of the icon (face, animal, etc.)
            - center_x: x coordinate of the icon center
            - center_y: y coordinate of the icon center
        """
        target_indices = np.where(self.__is_target_icon)
        target_data = pd.DataFrame({
            "icon_path": self.__icon_paths[target_indices],
            "icon_category": self.__icon_categories[target_indices],
            "center_x": self.__icon_centers[target_indices][:, 0],
            "center_y": self.__icon_centers[target_indices][:, 1]
        })
        return target_data

    @staticmethod
    def __identify_stimulus_type(stim_type) -> LWSStimulusTypeEnum:
        """
        Returns the correct value of LWSStimulusTypeEnum based on the input.
        :raises ValueError: if stim_type is not a valid value.
        """
        if isinstance(stim_type, LWSStimulusTypeEnum):
            return stim_type
        return LWSStimulusTypeEnum(stim_type.lower())

    def __eq__(self, other):
        if not isinstance(other, LWSArrayStimulus):
            return False
        return self.stim_id == other.stim_id and self.stim_type == other.stim_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.stim_type.name.upper()}{self.stim_id}"

    def __str__(self) -> str:
        return self.__repr__()
