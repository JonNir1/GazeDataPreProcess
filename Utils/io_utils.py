import os
from typing import Optional, Union

import experiment_config as cnfg


def create_subject_output_directory(subject_id: Union[int, str], output_dir: str = cnfg.OUTPUT_DIR) -> str:
    """
    create subject output directory with a unique name with the following format: PATH/TO/OUTPUT_DIR/SXXX
    """
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
        name, extension = split_name
    if extension is None:
        raise ValueError(f'No extension provided for filename: {name}')
    return f"{name}.{extension}"





