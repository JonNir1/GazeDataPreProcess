import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
from enum import StrEnum
from typing import Tuple

from Config import experiment_config as cnfg


class LWSStimulusTypeEnum(StrEnum):
    BW = 'bw'
    COLOR = 'color'
    NOISE = 'noise'


class LWSArrayStimulus:
    """
    This class represents a single LWS icon-array stimulus:
        - the stimulus image (as np.ndarray)
        - pixel location of icon centers
        - full paths to icon files that comprise the stimulus (image array)
        - categories of icon in the stimulus (faces, animals, etc.)
        - whether each icon is a target image
    """

    def __init__(self, stim_id: int, stim_type, image: np.ndarray,
                 icon_paths: np.ndarray, icon_centers: np.ndarray,
                 icon_categories: np.ndarray, is_target_icon: np.ndarray):
        self.__stim_id = stim_id
        self.__stim_type = self.__identify_stimulus_type(stim_type)
        self.__image = image  # color image in BGR format
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
        if stim_type == LWSStimulusTypeEnum.BW:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        f = loadmat(metadata_path)
        mat = f["imageInfo"]
        icon_paths = np.vectorize(lambda arr: arr[0])(mat["stimInArray"][0][0])  # shape (r, c)
        icon_centers = np.stack(np.vectorize(lambda arr: tuple(arr[0]))(mat["stimCenters"][0][0]), axis=2)  # shape (r, c, 2)
        icon_categories = mat["categoryInArray"][0][0].astype(int)  # shape (r, c)
        is_target_icon = mat["targetsInArray"][0][0].astype(bool)  # shape (r, c)

        return LWSArrayStimulus(stim_id, stim_type, image, icon_paths, icon_centers, icon_categories, is_target_icon)

    @staticmethod
    def from_type_and_id(stim_id: int, stim_type: str, stim_directory: str = cnfg.STIMULI_DIR) -> "LWSArrayStimulus":
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

    @property
    def image_shape(self) -> Tuple[int, int]:
        # returns the height & width of the stimulus image
        h = self.__image.shape[0]
        w = self.__image.shape[1]
        return h, w

    @property
    def icons_shape(self) -> Tuple[int, int]:
        # returns the number of rows & columns of icons in the stimulus
        return self.__is_target_icon.shape

    def get_image(self, color_format: str = 'bgr') -> np.ndarray:
        """
        Returns the stimulus image in the specified color format, default is BGR.
        The returned image is a copy of the original image so that the original image is not modified.
        :param color_format: 'bgr', 'rgb', or 'gray'
        :raises ValueError: if the color format is invalid
        """
        color_format = color_format.lower()
        if color_format == 'bgr':
            img = self.__image
        elif color_format == 'rgb':
            img = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)
        elif color_format == 'gray' or color_format == 'grey':
            img = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Invalid color format: {color_format}")
        return img.copy()

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
            "center_x": self.__icon_centers[target_indices][:, 1],  # X is the column index
            "center_y": self.__icon_centers[target_indices][:, 0]   # Y is the row index
        })
        return target_data

    def show(self, show_targets: bool = False):
        """
        Displays the stimulus image. Includes red circles around the target icons if show_targets is True.
        """
        im = self.get_image(color_format='bgr')
        if show_targets:
            target_data = self.get_target_data()
            for i, row in target_data.iterrows():
                x, y = int(row['center_x']), int(row['center_y'])
                cv2.circle(im, (x, y), 40, (0, 0, 255), 5)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

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
