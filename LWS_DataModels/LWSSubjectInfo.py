import os
import io
import datetime
import pandas as pd
from typing import Optional, List

from LWS_DataModels.LWSEnums import LWSSubjectSexEnum, LWSSubjectDominantHandEnum, LWSSubjectDominantEyeEnum


class LWSSubjectInfo:
    """
    This class is used to store subject metadata for a single subject in the LWS Demo experiment.
    """

    # Default E-Prime field names:
    SUBJECT_ID = "Subject"
    SESSION_ID = "Session"
    NAME = "Name"
    AGE = "Age"
    Sex = "Sex"
    DOMINANT_HAND = "Handedness"
    DOMINANT_EYE = "DominantEye"
    DISTANCE_TO_SCREEN = "Distance"
    Date = "SessionDate"
    Time = "SessionTime"

    def __init__(self, subject_id: int, session: Optional[int],
                 name: Optional[str], age: int, distance_to_screen: float,
                 date_and_time: datetime, sex: LWSSubjectSexEnum, dominant_hand: LWSSubjectDominantHandEnum,
                 dominant_eye: LWSSubjectDominantEyeEnum):
        self.__subject_id = subject_id
        self.__session = session if session is not None else 1
        self.__name = str(name)
        self.__age = age
        self.__distance_to_screen = distance_to_screen
        self.__date_and_time = date_and_time
        self.__sex = sex
        self.__dominant_eye = dominant_eye
        self.__dominant_hand = dominant_hand

    @staticmethod
    def from_eprime_file(fullpath: str) -> "LWSSubjectInfo":
        """
        This method is used to create a new LWSSubjectInfo object from an E-Prime file.
        :param fullpath: The E-Prime file to read from.
        :return: A new LWSSubjectInfo object.

        :raises FileNotFoundError: If the E-Prime file does not exist.
        :raises ValueError: If the specified file is not a subject-info E-Prime file.
        :raises ValueError: If the E-Prime file does not contain a mandatory field:
            subject id, age, sex, hand, eye, distance, date, time.
        """
        # check if the file exists and is a subject info file:
        if not os.path.isfile(fullpath):
            raise FileNotFoundError(f"The E-Prime file {fullpath} does not exist.")

        # extract all fields as strings from the E-Prime file:
        f = io.open(fullpath, mode="r", encoding="utf-16")
        lines = f.readlines()

        subject_id = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.SUBJECT_ID)
        session = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.SESSION_ID)
        name = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.NAME)
        age = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.AGE)
        distance_to_screen = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.DISTANCE_TO_SCREEN)
        date = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.Date)
        time = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.Time)
        sex = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.Sex)
        dominant_hand = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.DOMINANT_HAND)
        dominant_eye = LWSSubjectInfo.__extract_field(lines, LWSSubjectInfo.DOMINANT_EYE)
        f.close()

        # convert the fields to the correct types:
        if subject_id is None:
            raise ValueError(f"The E-Prime file does not contain a subject ID.")
        subject_id = int(subject_id)

        if age is None:
            raise ValueError(f"The E-Prime file does not contain an age.")
        age = int(age)

        if distance_to_screen is None:
            raise ValueError(f"The E-Prime file does not contain a distance to screen.")
        distance_to_screen = float(distance_to_screen)

        if date is None or time is None:
            raise ValueError(f"The E-Prime file does not contain a date and time.")
        date_and_time = datetime.datetime.strptime(f"{date} {time}", "%m-%d-%Y %H:%M:%S")

        if sex is None:
            raise ValueError(f"The E-Prime file does not contain a sex.")
        sex = LWSSubjectSexEnum[sex.capitalize()]

        if dominant_hand is None:
            raise ValueError(f"The E-Prime file does not contain a dominant hand.")
        dominant_hand = LWSSubjectDominantHandEnum[dominant_hand.capitalize()]

        if dominant_eye is None:
            raise ValueError(f"The E-Prime file does not contain a dominant eye.")
        dominant_eye = LWSSubjectDominantEyeEnum[dominant_eye.capitalize()]

        # these are not mandatory fields:
        session = int(session) if session is not None else None
        name = str(name).capitalize() if name is not None else None

        # create and return the new LWSSubjectInfo object:
        subject_info = LWSSubjectInfo(subject_id=subject_id, session=session, name=name, age=age,
                                      distance_to_screen=distance_to_screen, date_and_time=date_and_time, sex=sex,
                                      dominant_hand=dominant_hand, dominant_eye=dominant_eye)
        return subject_info

    @staticmethod
    def from_series(s: pd.Series) -> "LWSSubjectInfo":
        """
        Creates a new LWSSubjectInfo object from a Pandas Series.
        """
        subject_id = s["SubjectID"]
        session = s["Session"]
        name = s["Name"]
        age = s["Age"]
        distance_to_screen = s["DistanceToScreen"]
        date_and_time = s["DateTime"]
        sex = LWSSubjectSexEnum[s["Sex"].capitalize()]
        dominant_hand = LWSSubjectDominantHandEnum[s["DominantHand"].capitalize()]
        dominant_eye = LWSSubjectDominantEyeEnum[s["DominantEye"].capitalize()]
        subject_info = LWSSubjectInfo(subject_id=subject_id, session=session, name=name, age=age,
                                      distance_to_screen=distance_to_screen, date_and_time=date_and_time, sex=sex,
                                      dominant_hand=dominant_hand, dominant_eye=dominant_eye)
        return subject_info

    @property
    def subject_id(self) -> int:
        return self.__subject_id

    @property
    def session(self) -> int:
        return self.__session

    @property
    def name(self) -> str:
        return self.__name

    @property
    def age(self) -> int:
        return self.__age

    @property
    def distance_to_screen(self) -> float:
        return self.__distance_to_screen

    @property
    def date_time(self) -> datetime:
        return self.__date_and_time

    @property
    def sex(self) -> str:
        return self.__sex.name.capitalize()

    @property
    def dominant_hand(self) -> str:
        return self.__dominant_hand.name.capitalize()

    @property
    def dominant_eye(self) -> str:
        return self.__dominant_eye.name.capitalize()

    def to_series(self) -> pd.Series:
        """
        Converts the subject info to a pandas series.
        """
        return pd.Series(
            {"SubjectID": self.subject_id, "Session": self.session, "Name": self.name, "Age": self.age,
             "DistanceToScreen": self.distance_to_screen,
             "DateTime": pd.to_datetime(self.date_time.strftime("%Y-%m-%d %H:%M:%S")),
             "Sex": self.sex, "DominantHand": self.dominant_hand, "DominantEye": self.dominant_eye})

    @staticmethod
    def __extract_field(lines: List[str], field_name: str) -> Optional[str]:
        """
        Extracts a specified field from an E-Prime txt file by finding the line that starts with the field name and
        returning the value after the colon.
        :param lines:
        :param field_name:
        :return: the value of the field, or None if the field was not found.
        """
        field_name = field_name if field_name.endswith(":") else field_name + ":"
        lines_with_field_name = list(
            filter(lambda line: line.startswith(field_name) or line.startswith(field_name.capitalize()), lines)
        )
        if len(lines_with_field_name) == 0:
            return None
        first_line = lines_with_field_name[0]
        colon_index = first_line.find(":")
        if colon_index == -1:
            return None
        return first_line[colon_index + 1:].strip().capitalize()

    def __eq__(self, other):
        if not isinstance(other, LWSSubjectInfo):
            return False
        s1 = self.to_series()
        s2 = other.to_series()
        return s1.equals(s2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.subject_id}-{self.session}"

    def __str__(self) -> str:
        return self.__repr__()
