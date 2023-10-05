"""
This file contains the configuration for each specific experiment.
"""
import os
import numpy as np

from Config.ScreenMonitor import ScreenMonitor
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum

# DIRECTORIES
BASE_DIR = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo"
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# GLOBAL VARIABLES
ADDITIONAL_COLUMNS = ["ConditionName", "BlockNum", "TrialNum", "ImageNum"]  # additional columns to be added to the gaze data
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_default()  # global variable: screen monitor object
VIEWER_DISTANCE = 65  # global variable: distance between subject and screen center (cm)

# STIMULUS CONFIGURATIONS
# taken from the previously generated stimuli (see ExpInfo.mat and ArrayInfo.mat)
STIMULUS_BOTTOM_STRIP_TOP_LEFT = (730, 915)  # top left corner of the bottom strip of the stimulus
STIMULUS_BOTTOM_STRIP_TOP_RIGHT = (1190, 915)  # top right corner of the bottom strip of the stimulus
STIMULUS_BOTTOM_STRIP_BOTTOM_RIGHT = (STIMULUS_BOTTOM_STRIP_TOP_RIGHT[0], SCREEN_MONITOR.resolution[1])  # bottom right corner of the bottom strip of the stimulus

# GAZE DATA & GAZE EVENTS CONFIGURATION
DEFAULT_MINIMUM_SAMPLES_PER_EVENT = 2  # minimum number of samples in an event (saccade, fixation, etc.)
DEFAULT_INTER_EVENT_TIME = 5  # minimal time between two (same) events in milliseconds (two saccades, two fixations, etc.)
DEFAULT_MISSING_VALUE = np.nan  # default value indicating missing data in the gaze data (x, y)

DEFAULT_BLINK_MINIMUM_DURATION = 50  # minimum duration of a blink in milliseconds

DEFAULT_SACCADE_MINIMUM_DURATION = 5  # minimum duration of a saccade in milliseconds

DEFAULT_FIXATION_MINIMUM_DURATION = 55  # minimum duration of a fixation in milliseconds
DEFAULT_FIXATION_MAX_VELOCITY = 20  # degrees per second

# LWS ANALYSIS CONFIGURATION
THRESHOLD_VISUAL_ANGLE = 1.5  # threshold for the visual angle between a target and a gaze datapoint, in degrees
TARGET_IDENTIFICATION_SEQUENCE = np.array([ExperimentTriggerEnum.MARK_TARGET_SUCCESSFUL,
                                           ExperimentTriggerEnum.NULL,
                                           ExperimentTriggerEnum.CONFIRM_TARGET_SUCCESSFUL,
                                           ExperimentTriggerEnum.NULL])  # sequence of triggers indicating that a target was identified by the subject (in order)
PROX_THRESHOLDS = np.arange(THRESHOLD_VISUAL_ANGLE / 15,
                            26 * THRESHOLD_VISUAL_ANGLE / 15,
                            THRESHOLD_VISUAL_ANGLE / 15)    # proximity thresholds for identifying LWS instances
TIME_DIFF_THRESHOLDS = np.array([0, 10, 20, 50, 100, 200])  # Δt Threshold for identifying LWS instances
