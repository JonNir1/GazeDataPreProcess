"""
This file contains the configuration for each specific experiment.
"""

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

MISSING_VALUE = -1
SAMPLING_RATE = 599.88

ADDITIONAL_COLUMNS = []  # additional columns to be added to the gaze data


INTER_EVENT_TIME = 5  # minimal time between two (same) events in milliseconds (two saccades, two fixations, etc.)
BLINK_MINIMUM_DURATION = 50  # minimum duration of a blink in milliseconds
SACCADE_MINIMUM_DURATION = 5  # minimum duration of a saccade in milliseconds
FIXATION_MINIMUM_DURATION = 55  # minimum duration of a fixation in milliseconds
