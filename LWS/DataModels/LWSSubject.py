from typing import List

from LWS.DataModels.LWSSubjectInfo import LWSSubjectInfo


class LWSSubject:
    """
    Represents a single LWS subject, with it's personal info and experimental data
    """

    def __init__(self, info: LWSSubjectInfo, trials: List["LWSTrial"] = None):
        self.__subject_info: LWSSubjectInfo = info
        self.__trials: List[LWSTrial] = trials if trials is not None else []
        self.__is_processed = False

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
        return self.__is_processed

    @is_processed.setter
    def is_processed(self, is_processed: bool):
        if self.__is_processed and not is_processed:
            raise RuntimeError("Cannot set is_processed to False after it has been set to True.")
        self.__is_processed = is_processed

    @property
    def num_trials(self) -> int:
        return len(self.__trials)

    def get_trial(self, trial_num: int) -> "LWSTrial":
        trials = list(filter(lambda t: t.trial_num == trial_num, self.__trials))
        if len(trials) == 0:
            raise IndexError(f"Trial {trial_num} does not exist for Subject {self.subject_id}")
        if len(trials) > 1:
            raise RuntimeError(f"Subject {self.subject_id} has more than one trial with number {trial_num}")
        return trials[0]

    def add_trial(self, trial: "LWSTrial"):
        self.__trials.append(trial)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.subject_id:03d}"

    def __str__(self) -> str:
        return self.__repr__()

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



