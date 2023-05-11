import os
from typing import Optional

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.scripts.read_subject import read_subject
from LWS.scripts.detect_events import detect_all_events


def process_subject(subject_dir: str, stimuli_dir: str = cnfg.STIMULI_DIR,
                    screen_monitor: Optional[ScreenMonitor] = None, **kwargs):
    if not os.path.isdir(subject_dir):
        raise NotADirectoryError(f"Directory {subject_dir} does not exist.")
    if not os.path.isdir(stimuli_dir):
        raise NotADirectoryError(f"Directory {stimuli_dir} does not exist.")
    screen_monitor = screen_monitor if screen_monitor is not None else ScreenMonitor.from_config()
    sr, trials = read_subject(subject_dir, stimuli_dir, screen_monitor)
    for i, trial in enumerate(trials):
        is_blink, is_saccade, is_fixation = detect_all_events(trial, sr, **kwargs)

    pass
