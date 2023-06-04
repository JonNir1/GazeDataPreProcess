"""
This file contains the configuration for each specific experiment.
"""
import os
import numpy as np

# DIRECTORIES
BASE_DIR = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo"
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# GENERAL CONFIGURATION
SCREEN_DISTANCE = 65  # distance between the screen and the participant in cm
SCREEN_WIDTH = 53.5   # width of the screen in cm
SCREEN_HEIGHT = 31    # height of the screen in cm
SCREEN_RESOLUTION = (1920, 1080)  # resolution of the screen in pixels
SCREEN_REFRESH_RATE = 60  # refresh rate of the screen in Hz


# GAZE DATA & GAZE EVENTS CONFIGURATION
ADDITIONAL_COLUMNS = ["ConditionName", "BlockNum", "TrialNum", "ImageNum"]  # additional columns to be added to the gaze data
DEFAULT_INTER_EVENT_TIME = 5  # minimal time between two (same) events in milliseconds (two saccades, two fixations, etc.)
DEFAULT_MISSING_VALUE = np.nan  # default value indicating missing data in the gaze data (x, y)

DEFAULT_BLINK_MINIMUM_DURATION = 50  # minimum duration of a blink in milliseconds

DEFAULT_SACCADE_MINIMUM_DURATION = 5  # minimum duration of a saccade in milliseconds

DEFAULT_FIXATION_MINIMUM_DURATION = 55  # minimum duration of a fixation in milliseconds
DEFAULT_FIXATION_MAX_VELOCITY = 20  # degrees per second

# TRIGGERS CONFIGURATION
START_TRIGGER = 254  # trigger indicating the start of a trial
END_TRIGGER = 255  # trigger indicating the end of a trial


# ANALYSIS CONFIGURATION
THRESHOLD_VISUAL_ANGLE = 1.5  # threshold for the visual angle between a target and a gaze datapoint, in degrees
