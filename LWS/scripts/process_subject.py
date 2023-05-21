import os
from typing import Optional

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.scripts.read_subject import read_subject_trials
from LWS.scripts.detect_events import detect_all_events


def process_subject(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR, **kwargs):
    """

    :param subject_dir:
    :param stimuli_dir:

    :keyword screen_monitor:
    :keyword experiment_columns or additional_columns:
    :keyword start_trigger, end_trigger:


    :return:
    """

    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")
    kwargs["screen_monitor"] = kwargs.get("screen_monitor", None) or ScreenMonitor.from_config()

    trials = read_subject_trials(subject_dir, stimuli_dir, **kwargs)
    for i, trial in enumerate(trials):
        is_blink, is_saccade, is_fixation = detect_all_events(trial, sr, **kwargs)

    pass
