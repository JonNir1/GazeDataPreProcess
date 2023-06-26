import numpy as np
from scipy.ndimage import gaussian_filter
import warnings as warn  # to suppress numpy warnings
from typing import Optional
from collections import Counter

import constants as cnst
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.FixationEvent import FixationEvent


def gaze_heatmap(trial: LWSTrial, smoothing_sigma: Optional[float] = None) -> np.ndarray:
    """
    Returns a 2D array with the same shape as the stimulus image, where each entry is the number of times the subject's
    gaze fell on that pixel. If `smoothing_sigma` is not None, then the counts are smoothed using a Gaussian kernel with
    standard deviation `smoothing_sigma`.

    :raises ValueError: if `smoothing_sigma` is zero or negative.
    """
    h, w = trial.get_stimulus().image_shape
    heatmap = np.zeros((h, w))
    _, x_gaze, y_gaze = trial.get_raw_gaze_coordinates(eye='dominant')
    with warn.catch_warnings():
        warn.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
        x_gaze = np.rint(x_gaze).astype(int)
        y_gaze = np.rint(y_gaze).astype(int)
    counts = Counter(zip(y_gaze, x_gaze))
    for (y, x), count in counts.items():
        if y < 0 or y >= h:
            continue
        if x < 0 or x >= w:
            continue
        heatmap[y, x] = count
    if (smoothing_sigma is None) or (not np.isfinite(smoothing_sigma)):
        return heatmap
    if smoothing_sigma <= 0:
        raise ValueError(f"argument `smoothing_sigma` must be positive, got {smoothing_sigma}")
    return gaussian_filter(heatmap, sigma=smoothing_sigma)


def fixations_heatmap(trial: LWSTrial) -> np.ndarray:
    """
    Returns a 2D array with the same shape as the stimulus image, where each pixel is colored according to the total
    duration of all fixations that fell on it. Each fixation adds a 2D Gaussian to the heatmap, with the center of the
    Gaussian at the fixation's center of mass, it's mean equal to the fixation's duration (in milliseconds), and the
    standard deviation of the Gaussian equal to the fixation's standard deviation (on the X and Y axes).
    """
    h, w = trial.get_stimulus().image_shape
    heatmap = np.zeros((h, w))
    fixations = trial.get_gaze_events(cnst.FIXATION)
    for fixation in fixations:
        fixation: FixationEvent
        center_x, center_y = fixation.center_of_mass
        std_x, std_y = fixation.std
        duration = fixation.duration

        gaus = np.zeros_like(heatmap)
        gaus[round(center_y), round(center_x)] = duration
        gaus = gaussian_filter(gaus, sigma=[std_y, std_x])
        heatmap += gaus
    return heatmap
