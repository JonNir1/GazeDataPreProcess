import os
import traceback
from datetime import datetime
from typing import Optional, Union

from Config import experiment_config as cnfg

TEXT_EXTENSION = 'txt'
IMAGE_EXTENSION = 'png'
VIDEO_EXTENSION = 'mp4'
PICKLE_EXTENSION = 'pkl'


def create_subject_output_directory(subject_id: Union[int, str], output_dir: Optional[str] = cnfg.OUTPUT_DIR) -> str:
    """
    Create subject output directory with a unique name with the following format: PATH/TO/OUTPUT_DIR/SXXX
    If argument `output_dir` is not provided, the default output directory from `experiment_config` is used.
    """
    if output_dir is None:
        output_dir = cnfg.OUTPUT_DIR
    if isinstance(subject_id, int):
        subject_id = f"{subject_id:03d}"
    return create_directory(dirname=f"S{subject_id}", parent_dir=output_dir)


def create_directory(dirname: str, parent_dir: str) -> str:
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    full_path = os.path.join(parent_dir, dirname)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def get_filename(name: str, extension: Optional[str] = None) -> str:
    split_name = name.split('.')
    if len(split_name) > 2:
        raise ValueError(f'Invalid filename: {name}')
    if len(split_name) == 2:
        if extension is not None:
            raise ValueError(f'File already has an extension: {extension[1]}')
        name, extension = split_name
    if extension is None:
        raise ValueError(f'No extension provided for filename: {name}')
    return f"{name}.{extension}"


def print_and_log(msg: str, log_file: Optional[str] = None) -> None:
    print(msg)
    try:
        if log_file is not None:
            if not log_file.endswith(TEXT_EXTENSION):
                log_file = get_filename(name=log_file, extension=TEXT_EXTENSION)
            with open(log_file, 'a') as f:
                current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                f.write(f"[{current_time}] {msg}\n")
    except Exception as e:
        trace = traceback.format_exc()
        print(f"\tFailed to write to log file: {e}\n\t{trace}\n")
