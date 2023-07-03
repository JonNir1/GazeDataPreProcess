import os
import pickle as pkl
from typing import List, Optional

import Utils.io_utils as ioutils
from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo


class LWSSubject:
    """
    Represents a single LWS subject, with it's personal info and experimental data
    """

    def __init__(self, info: LWSSubjectInfo, trials: List["LWSTrial"] = None):
        self.__subject_info: LWSSubjectInfo = info
        self.__trials: List[LWSTrial] = trials if trials is not None else []

    @staticmethod
    def from_pickle(pickle_path: str) -> "LWSSubject":
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Could not find pickle file: {pickle_path}")
        with open(pickle_path, "rb") as f:
            subject = pkl.load(f)
        if not isinstance(subject, LWSSubject):
            raise RuntimeError(f"Expected LWSSubject, got {type(subject)}")
        return subject

    @property
    def subject_id(self) -> int:
        return self.__subject_info.subject_id

    @property
    def dominant_eye(self) -> str:
        return self.__subject_info.dominant_eye

    @property
    def distance_to_screen(self) -> float:
        return self.__subject_info.distance_to_screen

    @property
    def is_processed(self) -> bool:
        return all([trial.is_processed for trial in self.__trials])

    @property
    def num_trials(self) -> int:
        return len(self.__trials)

    def add_trial(self, trial: "LWSTrial"):
        self.__trials.append(trial)

    def get_all_trials(self) -> List["LWSTrial"]:
        return self.__trials

    def get_trial(self, trial_num: int) -> "LWSTrial":
        trials = list(filter(lambda t: t.trial_num == trial_num, self.__trials))
        if len(trials) == 0:
            raise IndexError(f"Trial {trial_num} does not exist for Subject {self.subject_id}")
        if len(trials) > 1:
            raise RuntimeError(f"Subject {self.subject_id} has more than one trial with number {trial_num}")
        return trials[0]

    def to_pickle(self, output_dir: Optional[str] = None) -> str:
        subject_dir = ioutils.create_subject_output_directory(subject_id=self.subject_id, output_dir=output_dir)
        filename = ioutils.get_filename(name=self.__repr__(), extension=ioutils.PICKLE_EXTENSION)
        full_path = os.path.join(subject_dir, filename)
        with open(full_path, "wb") as f:
            pkl.dump(self, f)
        return full_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.subject_id:03d}"

    def __str__(self) -> str:
        return f"S{self.subject_id:03d}"

    def __eq__(self, other: "LWSSubject") -> bool:
        if not isinstance(other, LWSSubject):
            return False
        if self.subject_id != other.subject_id:
            return False
        if self.is_processed != other.is_processed:
            return False
        if self.__subject_info != other.__subject_info:
            return False
        if self.num_trials != other.num_trials:
            return False
        for i in range(self.num_trials):
            if self.__trials[i] != other.__trials[i]:
                return False
        return True


# import at the bottom to avoid circular imports
from LWS.DataModels.LWSTrial import LWSTrial



