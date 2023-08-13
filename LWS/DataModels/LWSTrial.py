import os
import numpy as np
import pickle as pkl
import warnings as w
from typing import Tuple, List, Optional

import pandas as pd

import constants as cnst
import Utils.io_utils as ioutils
import Utils.array_utils as au
from Config import experiment_config as cnfg
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum
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

    def get_targets(self, basic_only: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe containing information about the trial's targets.
        :param basic_only: if True, return only the basic information about the targets. If False, return additional
            information about the targets.

        Basic information:
            - icon_path: full path to the icon file
            - icon_category: category of the icon (face, animal, etc.)
            - center_x: x coordinate of the icon center
            - center_y: y coordinate of the icon center

        Additional information:
            - distance_identified: distance (in visual angle) between the target and the gaze when the target was
                identified by the subject
            - time_identified: time (in milliseconds) when the target was identified by the subject
            - time_confirmed: time (in milliseconds) when the target was confirmed by the subject
        """
        targets_df = self.__stimulus.get_target_data()
        if basic_only:
            return targets_df

        target_identification_data = self._extract_target_identification_data()
        final_df = pd.concat([targets_df, target_identification_data], axis=1)
        return final_df

    def get_behavioral_data(self) -> LWSBehavioralData:
        return self.__behavioral_data

    def set_behavioral_data(self, behavioral_data: LWSBehavioralData):
        if self.is_processed:
            raise RuntimeError("Cannot set behavioral data after trial has been processed.")
        self.__behavioral_data = behavioral_data

    def get_gaze_events(self,
                        event_type: Optional[GazeEventTypeEnum] = None,
                        ignore_outliers: bool = False) -> List[BaseGazeEvent]:
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
            events[(ev.start_time <= timestamps) & (timestamps <= ev.end_time)] = ev.event_type()
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

    def _extract_target_identification_data(self) -> pd.DataFrame:
        """
        For each of the trial's targets, extracts the following information:
            - distance_identified: distance (in visual angle) between the target and the gaze when the target was
                identified by the subject
            - time_identified: time (in milliseconds) when the target was identified by the subject
            - time_confirmed: time (in milliseconds) when the target was confirmed by the subject

        Returns a dataframe with shape (num_targets, 3), where each row corresponds to a target.
        """
        FULL_IDENTIFICATION_SEQUENCE = np.array([ExperimentTriggerEnum.MARK_TARGET_SUCCESSFUL,
                                                 ExperimentTriggerEnum.NULL,
                                                 ExperimentTriggerEnum.CONFIRM_TARGET_SUCCESSFUL,
                                                 ExperimentTriggerEnum.NULL])

        # extract relevant columns from the behavioral data
        behavioral_data = self.get_behavioral_data()
        columns = ([cnst.MICROSECONDS, cnst.TRIGGER, "closest_target"] +
                   [col for col in behavioral_data.columns if col.startswith(f"{cnst.DISTANCE}_{cnst.TARGET}")])
        behavioral_df = pd.DataFrame(behavioral_data.get(columns), columns=columns)

        res = pd.DataFrame(np.full((self.num_targets, 3), np.nan),
                           columns=["distance_identified", "time_identified", "time_confirmed"])
        for i in range(self.num_targets):
            proximal_behavioral_df = behavioral_df[behavioral_df["closest_target"] == i + 1]

            # check if target was ever identified by the subject
            identification_idxs = au.find_sequences_in_sparse_array(proximal_behavioral_df[cnst.TRIGGER].values,
                                                                    sequence=FULL_IDENTIFICATION_SEQUENCE)
            if len(identification_idxs) == 0:
                # this target was never identified
                continue

            # check if any of the target's identification attempts were from below the threshold distance
            identification_distances = np.array(
                [proximal_behavioral_df.iloc[first_idx][f"{cnst.DISTANCE}_{cnst.TARGET}{i + 1}"]
                 for first_idx, last_idx in identification_idxs])
            proximal_identifications = np.where(identification_distances < cnfg.THRESHOLD_VISUAL_ANGLE)[0]
            if len(proximal_identifications) == 0:
                # no proximal identification attempts
                continue

            # find the start & end idxs of the first identification attempt that was from below the threshold distance
            first_proximal_identification = min(proximal_identifications)
            first_proximal_identification_idxs = identification_idxs[first_proximal_identification]
            first_idx, last_idx = first_proximal_identification_idxs
            res.loc[i, "distance_identified"] = proximal_behavioral_df.iloc[first_idx][
                f"{cnst.DISTANCE}_{cnst.TARGET}{i + 1}"]
            res.loc[i, "time_identified"] = proximal_behavioral_df.iloc[first_idx][
                                                       cnst.MICROSECONDS] / cnst.MICROSECONDS_PER_MILLISECOND
            res.loc[i, "time_confirmed"] = proximal_behavioral_df.iloc[last_idx][
                                                      cnst.MICROSECONDS] / cnst.MICROSECONDS_PER_MILLISECOND
        return res

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


# import at the bottom to avoid circular imports
from LWS.DataModels.LWSSubject import LWSSubject
