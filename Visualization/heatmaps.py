import numpy as np
from scipy.ndimage import gaussian_filter
import warnings as warn  # to suppress numpy warnings
from collections import Counter
from typing import Tuple, List

import Utils.array_utils as au
from GazeEvents.FixationEvent import FixationEvent


def fixations_heatmap(fixations: List[FixationEvent], screen_resolution: Tuple[float, float]) -> np.ndarray:
    """
    Returns a 2D array with the same shape as the stimulus image, where each pixel is colored according to the total
    duration of all fixations that fell on it. Each fixation adds a 2D Gaussian to the heatmap, with the center of the
    Gaussian at the fixation's center of mass, it's mean equal to the fixation's duration (in milliseconds), and the
    standard deviation of the Gaussian equal to the fixation's standard deviation (on the X and Y axes).
    """
    w, h = screen_resolution
    if h <= 0 or w <= 0:
        raise ValueError(f"screen resolution must be positive, got {screen_resolution}")

    heatmap = np.zeros((h, w))
    for fixation in fixations:
        center_x, center_y = fixation.center_of_mass  # could be outside the screen
        if (not 0 <= center_x < w) or (not 0 <= center_y < h):
            continue

        duration = fixation.duration
        std_x, std_y = fixation.standard_deviation

        addition = np.zeros_like(heatmap)
        addition[round(center_y), round(center_x)] = duration
        addition = gaussian_filter(addition, sigma=(std_y, std_x))
        heatmap += addition
    # normalize heatmap to values in [0, 1]
    heatmap = au.normalize_array(heatmap)
    return heatmap


def gaze_heatmap(x_gaze: np.ndarray, y_gaze: np.ndarray,
                 screen_resolution: Tuple[float, float], smoothing_std: float) -> np.ndarray:
    """
    Returns a 2D array where each entry is the relative amount of samples that fell on the corresponding pixel.

    :param x_gaze: the x coordinates of the subject's gaze samples
    :param y_gaze: the y coordinates of the subject's gaze samples
    :param screen_resolution: the resolution of the screen on which the stimulus was presented (in pixels)
    :param smoothing_std: the standard deviation of the Gaussian used to smooth the heatmap

    :raises ValueError: if `screen_resolution` is non-positive.
    :raises ValueError: if `x_gaze` and `y_gaze` have different lengths.
    :raises ValueError: if `smoothing_sigma` is nan, inf, zero or negative.
    """
    if (smoothing_std is None) or (not np.isfinite(smoothing_std)):
        raise ValueError(f"argument `smoothing_std` must be a finite number, got {smoothing_std}")
    if smoothing_std <= 0:
        raise ValueError(f"argument `smoothing_sigma` must be positive, got {smoothing_std}")

    heatmap = __pixel_counts(x_gaze, y_gaze, screen_resolution)
    heatmap = gaussian_filter(heatmap, sigma=smoothing_std)
    # normalize heatmap to values in [0, 1]
    heatmap = au.normalize_array(heatmap)
    return heatmap


def __pixel_counts(x_gaze: np.ndarray, y_gaze: np.ndarray, screen_resolution: Tuple[float, float]) -> np.ndarray:
    """
    Returns a 2D array where each entry is the number of times the subject's gaze fell on the corresponding pixel.

    :param x_gaze: the x coordinates of the subject's gaze samples
    :param y_gaze: the y coordinates of the subject's gaze samples
    :param screen_resolution: the resolution of the screen on which the stimulus was presented (in pixels)

    :raises ValueError: if `screen_resolution` is non-positive.
    :raises ValueError: if `x_gaze` and `y_gaze` have different lengths.
    """
    w, h = screen_resolution
    if h <= 0 or w <= 0:
        raise ValueError(f"screen resolution must be positive, got {screen_resolution}")
    if len(x_gaze) != len(y_gaze):
        raise ValueError(f"arguments `x_gaze` and `y_gaze` must have the same length, got {len(x_gaze)} and {len(y_gaze)}")

    counts = np.zeros((h, w))
    with warn.catch_warnings():
        # cast gaze pixels to int, ignore warnings about NaNs
        warn.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
        x_gaze = np.rint(x_gaze).astype(int)
        y_gaze = np.rint(y_gaze).astype(int)
    counter = Counter(zip(y_gaze, x_gaze))
    for (y, x), c in counter.items():
        if y < 0 or y >= h:
            continue
        if x < 0 or x >= w:
            continue
        counts[y, x] = c
    return counts

