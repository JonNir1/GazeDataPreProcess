from typing import List, Union
from DataParser.BaseEyeTrackingParser import BaseEyeTrackingParser


class TobiiCSVEyeTrackingParser(BaseEyeTrackingParser):
    """
    Parses eye-tracking data based on the CSV format exported by Tobii eye-tracker and E-Prime.
    """

    @classmethod
    def MISSING_VALUES(cls) -> List[Union[float, str]]:
        return [-1, "-1", "-1.#IND0"]

    @classmethod
    def TRIAL_COLUMN(cls) -> str:
        return 'RunningSample'

    @classmethod
    def MILLISECONDS_COLUMN(cls) -> str:
        return 'RTTime'

    @classmethod
    def MICROSECONDS_COLUMN(cls) -> str:
        return 'RTTimeMicro'

    @classmethod
    def LEFT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXLeftEye'

    @classmethod
    def LEFT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYLeftEye'

    @classmethod
    def LEFT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterLeftEye"

    @classmethod
    def RIGHT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXRightEye'

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYRightEye'

    @classmethod
    def RIGHT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterRightEye"
