import os
import numpy as np
import pickle as pkl
import warnings as w
from typing import Tuple, List, Optional

import constants as cnst
import Utils.io_utils as ioutils
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSBehavioralData import LWSBehavioralData
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class LWSTrial:
    """
    Represents a single trial in the LWS Demo experiment.
    """

    def __init__(self,
                 trial_num: int,
                 subject_info: LWSSubjectInfo,
                 stimulus: LWSArrayStimulus,
                 behavioral_data: LWSBehavioralData,
                 gaze_events: List[BaseGazeEvent] = None):
        self.__is_processed: bool = False
        self.__trial_num: int = trial_num
        self.__subject_info: LWSSubjectInfo = subject_info
        self.__stimulus: LWSArrayStimulus = stimulus
        self.__behavioral_data: LWSBehavioralData = behavioral_data
        self.__gaze_events: List[BaseGazeEvent] = sorted(self.__gaze_events,
                                                         key=lambda e: e.start_time) if gaze_events is not None else []

    @property
    def trial_num(self) -> int:
        return self.__trial_num

    @property
    def sampling_rate(self) -> float:
        return self.__behavioral_data.sampling_rate

    @property
    def duration(self) -> float:
        # duration in milliseconds
        timestamps = self.__behavioral_data.get(cnst.MICROSECONDS).values
        return (timestamps[-1] - timestamps[0]) / cnst.MICROSECONDS_PER_MILLISECOND

    @property
    def is_processed(self) -> bool:
        return self.__is_processed

    @is_processed.setter
    def is_processed(self, is_processed: bool):
        if self.__is_processed and not is_processed:
            raise RuntimeError("Cannot set is_processed to False after it has been set to True.")
        self.__is_processed = is_processed

    def get_subject_info(self) -> LWSSubjectInfo:
        return self.__subject_info

    def get_stimulus(self) -> LWSArrayStimulus:
        return self.__stimulus

    def get_behavioral_data(self) -> LWSBehavioralData:
        return self.__behavioral_data

    def set_behavioral_data(self, behavioral_data: LWSBehavioralData):
        if self.is_processed:
            raise RuntimeError("Cannot set behavioral data after trial has been processed.")
        self.__behavioral_data = behavioral_data

    def get_gaze_events(self, event_type: Optional[str]) -> List[BaseGazeEvent]:
        if self.__gaze_events is None:
            return []
        if len(self.__gaze_events) == 0:
            return []
        if event_type is None:
            return self.__gaze_events

        event_type = event_type.lower()
        if event_type == cnst.ALL:
            return self.__gaze_events
        return list(filter(lambda e: e.event_type() == event_type, self.__gaze_events))

    def set_gaze_events(self, gaze_events: List[BaseGazeEvent]):
        if self.is_processed:
            raise RuntimeError("Cannot set gaze events after trial has been processed.")
        ge = self.get_gaze_events(event_type=cnst.ALL)
        if len(ge) > 0:
            w.warn("Overwriting existing gaze events.")
        self.__gaze_events = sorted(gaze_events, key=lambda e: e.start_time)

    def get_raw_gaze_coordinates(self, eye: str = 'dominant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the raw gaze coordinates for the given eye or both eyes, along with the timestamps.
        :param eye: controls which eye's gaze coordinates are returned:
            - if 'dominant', return the dominant eye's gaze coordinates (as defined in the subject info)
            - if 'left', return left eye's gaze coordinates
            - if 'right', return right eye's gaze coordinates
            - if 'both', return gaze coordinates from both eyes (as a 2xN array)
            - otherwise, raise a ValueError

        :return: a tuple of (timestamps, x coordinates, y coordinates)
        """
        # Returns the timestamp, x and y coordinates of the gaze data for the dominant eye.
        bd = self.get_behavioral_data()
        ts = bd.get(cnst.MICROSECONDS).values / 1000

        eye = eye.lower()
        if eye == "dominant":
            eye = self.get_subject_info().dominant_eye.lower()
        if eye == 'left':
            x_l, y_l = bd.get(cnst.LEFT_X).values, bd.get(cnst.LEFT_Y).values
            return ts, x_l, y_l
        if eye == 'right':
            x_r, y_r = bd.get(cnst.RIGHT_X).values, bd.get(cnst.RIGHT_Y).values
            return ts, x_r, y_r
        if eye == 'both':
            x_l, y_l = bd.get(cnst.LEFT_X).values, bd.get(cnst.LEFT_Y).values
            x_r, y_r = bd.get(cnst.RIGHT_X).values, bd.get(cnst.RIGHT_Y).values
            return ts, np.vstack((x_l, x_r)), np.vstack((y_l, y_r))
        raise ValueError(f'Invalid eye: {eye}')

    def to_pickle(self, output_dir: Optional[str] = None) -> str:
        subject_dir = ioutils.create_subject_output_directory(subject_id=self.__subject_info.subject_id,
                                                              output_dir=output_dir)
        trials_dir = ioutils.create_directory(dirname='trials', parent_dir=subject_dir)
        full_path = os.path.join(trials_dir, f"{str(self)}.pkl")
        with open(full_path, "wb") as f:
            pkl.dump(self, f)
        return full_path

    @staticmethod
    def from_pickle(pickle_path: str) -> 'LWSTrial':
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Could not find pickle file: {pickle_path}")
        with open(pickle_path, "rb") as f:
            trial = pkl.load(f)
        if not isinstance(trial, LWSTrial):
            raise RuntimeError(f"Expected LWSTrial, got {type(trial)}")
        return trial

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_S{self.__subject_info.subject_id}_T{self.__trial_num}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, LWSTrial):
            return False
        if not self.__trial_num == other.__trial_num:
            return False
        if not self.__is_processed == other.__is_processed:
            return False

        self_subject_info = self.get_subject_info()
        other_subject_info = other.get_subject_info()
        if not self_subject_info == other_subject_info:
            return False

        self_bdata = self.get_behavioral_data()
        other_bdata = other.get_behavioral_data()
        if not self_bdata == other_bdata:
            return False

        self_gaze_events = self.get_gaze_events(event_type=cnst.ALL)
        other_gaze_events = other.get_gaze_events(event_type=cnst.ALL)
        if not len(self_gaze_events) == len(other_gaze_events):
            return False
        for i in range(len(self_gaze_events)):
            if not self_gaze_events[i] == other_gaze_events[i]:
                return False
        return True
