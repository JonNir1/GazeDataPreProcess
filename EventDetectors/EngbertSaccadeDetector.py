import numpy as np

from EventDetectors.BaseSaccadeDetector import BaseSaccadeDetector


class EngbertSaccadeDetector(BaseSaccadeDetector):
    """
    Saccade detector based on the algorithm described by Engbert, Kliegl, and Mergenthaler (2003, 2006).
    See implementations in the following repositories:
        - https://github.com/Yuvishap/Gaze-Project/blob/67acb26fc90e5148a05b47ca7711306c94b79ed7/Gaze/src/pre_processing/business/EngbertFixationsLogic.py
        - https://github.com/odedwer/EyelinkProcessor/blob/66f56463ba8d2ad75f7935e3d020b051fb2aa4a4/SaccadeDetectors.py
        - https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py#L175
    """

    def detect(self, gaze_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

