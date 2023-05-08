import os
import re
from typing import List


def find_data_files_by_suffix(directory: str, end_with: str) -> List[str]:
    """
    Find all files in a directory that end with a specific string and match the e-prime naming convention:
        "<ExpName>-<SubjectID>-<Session>-<DataType>.txt"
    examples:
        - Subject Info from E-Prime: "ExpName-21-33.txt"
        - GazeData from Tobii: "ExpName-21-33-GazeData.txt"
        - TriggerLog from E-Prime: "ExpName-21-33-Trigger-Log.txt"
    """
    if end_with:
        end_with = f"-{end_with}.txt"
    else:
        end_with = ".txt"
    pattern = re.compile("[a-zA-z0-9]*-[0-9]*-[0-9]*" + end_with)
    paths = [os.path.join(directory, file) for file in os.listdir(directory) if pattern.match(file)]
    paths.sort(key=lambda x: _extract_session_from_filename(x))
    return paths


def _extract_session_from_filename(filename: str) -> int:
    # from a filename like "ExpName-21-33-SomeText.txt" extract the last number (session number; 33 in this case)
    return int(filename.split(".")[0].split("-")[2])
