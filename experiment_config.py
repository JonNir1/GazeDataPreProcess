"""
This file contains the configuration for each specific experiment.
"""

SCREEN_DISTANCE = 65  # distance between the screen and the participant in cm
SCREEN_WIDTH = 53.5   # width of the screen in cm
SCREEN_HEIGHT = 31    # height of the screen in cm
SCREEN_RESOLUTION = (1920, 1080)  # resolution of the screen in pixels
SCREEN_REFRESH_RATE = 60  # refresh rate of the screen in Hz

# ADDITIONAL_COLUMNS = []  # additional columns to be added to the gaze data
# STIMULI_DIR = ""  # directory containing the stimuli

# LWS SPECIFIC CONFIGURATION
ADDITIONAL_COLUMNS = ["ConditionName", "BlockNum", "TrialNum", "ImageNum"]
STIMULI_DIR = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\Stimuli\generated_stim1"
