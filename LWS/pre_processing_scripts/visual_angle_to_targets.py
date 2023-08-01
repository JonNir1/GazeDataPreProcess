# LWS PreProcessing Pipeline

import numpy as np
import pandas as pd
from typing import List

import Config.experiment_config as cnfg
import Utils.angle_utils as angle_utils
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def visual_angle_gaze_to_targets(trial: LWSTrial) -> np.ndarray:
    """
    Calculates the visual angle between each gaze datapoint in the trial, and each of the trial's target.
    Fills with np.inf for gaze-points with missing or invalid data.
    Returns a 2D array of shape (num_targets, num_gaze_points).
    """
    _ts, xs, ys, _ps = trial.get_raw_gaze_data(eye='dominant')
    target_info = trial.get_targets()
    angular_distances = np.zeros((target_info.shape[0], len(xs)))
    for i, row in target_info.iterrows():
        tx, ty = row['center_x'], row['center_y']
        dists = _calculate_visual_angle_to_target(tx, ty, xs, ys, trial.subject.distance_to_screen)
        angular_distances[i] = dists
    return angular_distances


def visual_angle_fixation_to_targets(fix: LWSFixationEvent, trial: LWSTrial) -> List[float]:
    """
    Calculates the visual angle between the fixation's center-of-mass, and each of the trial's targets.
    Returns np.inf if the fixation event's center-of-mass is missing or invalid.
    """
    fix_x, fix_y = fix.center_of_mass
    target_info = trial.get_targets()

    angular_distances = []
    for _i, row in target_info.iterrows():
        tx, ty = row['center_x'], row['center_y']
        dist = _calculate_visual_angle_to_target(tx, ty, np.array([fix_x]), np.array([fix_y]), trial.subject.distance_to_screen)
        angular_distances.append(dist[0])
    return angular_distances


def _calculate_visual_angle_to_target(target_x: float, target_y: float,
                                      xs: np.ndarray, ys: np.ndarray,
                                      viewer_distance: float) -> np.ndarray:
    """
    Calculate the visual angle (in degrees) between the target and each of the points specified by xs, ys.
    :param tx, target_y: target x, y coordinates
    :param xs, ys: arrays of (x, y) coordinates
    """
    if len(xs) != len(ys):
        raise ValueError("x and y must be of the same length")
    if np.isnan(target_x) or np.isnan(target_y):
        raise ValueError("Target coordinates cannot be NaN")
    if not np.isfinite(viewer_distance):
        raise ValueError("Viewer distance must be finite")
    if viewer_distance <= 0:
        raise ValueError("Viewer distance must be positive")

    pixel_size = cnfg.SCREEN_MONITOR.pixel_size
    distances = np.zeros_like(xs)
    for i in range(len(xs)):
        distances[i] = angle_utils.calculate_visual_angle(p1=(target_x, target_y), p2=(xs[i], ys[i]),
                                                          d=viewer_distance, pixel_size=pixel_size,
                                                          use_radians=False)
    return distances


def _calculate_euclidean_distance_to_target(tx: float, ty: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Returns the Euclidean distance (in pixels) between the target and each of the gaze points.
    :param tx, ty: target x, y coordinates
    :param xs, ys: arrays of gaze x, y coordinates
    """
    if len(xs) != len(ys):
        raise ValueError("x and y must be of the same length")
    if np.isnan(tx) or np.isnan(ty):
        raise ValueError("Target coordinates cannot be NaN")

    if len(xs) == 0:
        return np.array([])
    return np.sqrt((tx - xs) ** 2 + (ty - ys) ** 2)
