import os
import numpy as np
import pickle as pkl
import warnings as w
from typing import Tuple, List, Optional

import pandas as pd

import constants as cnst
import Utils.io_utils as ioutils
from LWS.DataModels.LWSArrayStimulus import LWSArrayStimulus
from LWS.DataModels.LWSBehavioralData import LWSBehavioralData
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class LWSTrial:
    """
    Represents a single trial in the LWS Demo experiment.
    """

    def __init__(self,
                 trial_num: int,
                 stimulus: LWSArrayStimulus,
                 behavioral_data: LWSBehavioralData,
                 gaze_events: List[BaseGazeEvent] = None,
                 subject: "LWSSubject" = None):
        self.__is_processed: bool = False
        self.__trial_num: int = trial_num
        self.__stimulus: LWSArrayStimulus = stimulus
        self.__behavioral_data: LWSBehavioralData = behavioral_data
        self.__gaze_events: List[BaseGazeEvent] = sorted(self.__gaze_events,
                                                         key=lambda e: e.start_time) if gaze_events is not None else []
        self.__subject: LWSSubject = subject

    @staticmethod
    def from_pickle(pickle_path: str) -> 'LWSTrial':
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Could not find pickle file: {pickle_path}")
        with open(pickle_path, "rb") as f:
            trial = pkl.load(f)
        if not isinstance(trial, LWSTrial):
            raise RuntimeError(f"Expected LWSTrial, got {type(trial)}")
        return trial

    @property
    def subject(self) -> "LWSSubject":
        return self.__subject

    @property
    def trial_num(self) -> int:
        return self.__trial_num

    @property
    def num_samples(self) -> int:
        timestamps = self.__behavioral_data.get(cnst.MICROSECONDS)
        return len(timestamps)

    @property
    def num_targets(self) -> int:
        return self.__stimulus.num_targets

    @property
    def sampling_rate(self) -> float:
        return self.__behavioral_data.sampling_rate

    @property
    def start_time(self) -> float:
        # start time in milliseconds
        timestamps = self.__behavioral_data.get(cnst.MICROSECONDS)
        return timestamps[0] / cnst.MICROSECONDS_PER_MILLISECOND

    @property
    def end_time(self) -> float:
        # end time in milliseconds
        timestamps = self.__behavioral_data.get(cnst.MICROSECONDS)
        return timestamps[-1] / cnst.MICROSECONDS_PER_MILLISECOND

    @property
    def duration(self) -> float:
        # duration in milliseconds
        return self.end_time - self.start_time

    @property
    def is_processed(self) -> bool:
        return self.__is_processed

    @is_processed.setter
    def is_processed(self, is_processed: bool):
        if self.__is_processed and not is_processed:
            raise RuntimeError("Cannot set is_processed to False after it has been set to True.")
        self.__is_processed = is_processed

    def get_stimulus_image(self, color_format: str = 'bgr') -> np.ndarray:
        return self.__stimulus.get_image(color_format=color_format)

    def get_targets(self) -> pd.DataFrame:
        """
        Returns a dataframe containing information about the trial's targets:
            - icon_path: full path to the icon file
            - icon_category: category of the icon (face, animal, etc.)
            - center_x: x coordinate of the icon center
            - center_y: y coordinate of the icon center
        """
        return self.__stimulus.get_target_data()

    def get_behavioral_data(self) -> LWSBehavioralData:
        return self.__behavioral_data

    def set_behavioral_data(self, behavioral_data: LWSBehavioralData):
        if self.is_processed:
            raise RuntimeError("Cannot set behavioral data after trial has been processed.")
        self.__behavioral_data = behavioral_data

    def get_gaze_events(self,
                        event_type: Optional[GazeEventTypeEnum] = None,
                        ignore_outliers: bool = False) -> list:
        if self.__gaze_events is None:
            return []
        if len(self.__gaze_events) == 0:
            return []

        if event_type is None:
            gaze_events = self.__gaze_events
        else:
            gaze_events = list(filter(lambda e: e.event_type() == event_type, self.__gaze_events))
        if not ignore_outliers:
            return gaze_events
        return list(filter(lambda e: not e.is_outlier, gaze_events))

    def set_gaze_events(self, gaze_events: List[BaseGazeEvent]):
        if self.is_processed:
            raise RuntimeError("Cannot set gaze events after trial has been processed.")
        ge = self.get_gaze_events()
        if len(ge) > 0:
            w.warn("Overwriting existing gaze events.")
        self.__gaze_events = sorted(gaze_events, key=lambda e: e.start_time)

    def get_raw_gaze_data(self, eye: str = 'dominant') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the raw gaze coordinates for the given eye or both eyes, along with the timestamps and pupil sizes.
        :param eye: controls which eye's gaze coordinates are returned:
            - if 'dominant', return the dominant eye's gaze coordinates (as defined in the subject info)
            - if 'left', return left eye's gaze coordinates
            - if 'right', return right eye's gaze coordinates
            - if 'both', return gaze coordinates from both eyes (as a 2xN array)
            - otherwise, raise a ValueError

        :return: a tuple of (timestamps, x coordinates, y coordinates, pupil sizes)
        """
        bd = self.get_behavioral_data()
        ts = bd.get(cnst.MICROSECONDS) / 1000

        eye = eye.lower()
        if eye == "dominant":
            eye = self.subject.dominant_eye.lower()
        if eye == 'left':
            x_l = bd.get(cnst.LEFT_X)
            y_l = bd.get(cnst.LEFT_Y)
            p_l = bd.get(cnst.LEFT_PUPIL)
            return ts, x_l, y_l, p_l
        if eye == 'right':
            x_r = bd.get(cnst.RIGHT_X)
            y_r = bd.get(cnst.RIGHT_Y)
            p_r = bd.get(cnst.RIGHT_PUPIL)
            return ts, x_r, y_r, p_r
        if eye == 'both':
            x_l, y_l, p_l = bd.get(cnst.LEFT_X), bd.get(cnst.LEFT_Y), bd.get(cnst.LEFT_PUPIL)
            x_r, y_r, p_r = bd.get(cnst.RIGHT_X), bd.get(cnst.RIGHT_Y), bd.get(cnst.RIGHT_PUPIL)
            return ts, np.vstack((x_l, x_r)), np.vstack((y_l, y_r)), np.vstack((p_l, p_r))
        raise ValueError(f'Invalid eye: {eye}')

    def get_triggers(self) -> np.ndarray:
        """ Returns the trigger values for this trial. """
        return self.__behavioral_data.get(cnst.TRIGGER)

    def get_event_per_sample(self) -> List[GazeEventTypeEnum]:
        """
        Returns a list identifying each sample as belonging to a particular event, based on the trial's `gaze_events`.
        """
        timestamps, _, _, _ = self.get_raw_gaze_data()
        events = np.full(timestamps.shape, GazeEventTypeEnum.UNDEFINED)
        for ev in self.get_gaze_events():
            events[ev.start_time <= timestamps <= ev.end_time] = ev.event_type()
        return list(events)

    def to_pickle(self, output_dir: Optional[str] = None) -> str:
        subject_dir = ioutils.create_subject_output_directory(subject_id=self.subject.subject_id,
                                                              output_dir=output_dir)
        trials_dir = ioutils.create_directory(dirname='trials', parent_dir=subject_dir)
        filename = ioutils.get_filename(name=self.__repr__(), extension=ioutils.PICKLE_EXTENSION)
        full_path = os.path.join(trials_dir, filename)
        with open(full_path, "wb") as f:
            pkl.dump(self, f)
        return full_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{str(self.subject)}_{str(self)}"

    def __str__(self) -> str:
        return f"T{self.__trial_num:03d}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LWSTrial):
            return False
        if not self.__trial_num == other.__trial_num:
            return False
        if not self.__is_processed == other.__is_processed:
            return False

        self_bdata = self.get_behavioral_data()
        other_bdata = other.get_behavioral_data()
        if not self_bdata == other_bdata:
            return False

        self_gaze_events = self.get_gaze_events()
        other_gaze_events = other.get_gaze_events()
        if not len(self_gaze_events) == len(other_gaze_events):
            return False
        for i in range(len(self_gaze_events)):
            if not self_gaze_events[i] == other_gaze_events[i]:
                return False
        return True

    def __hash__(self):
        return hash(self.__repr__())


# import at the bottom to avoid circular imports
from LWS.DataModels.LWSSubject import LWSSubject
