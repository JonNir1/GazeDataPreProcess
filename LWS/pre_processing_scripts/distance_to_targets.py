# LWS PreProcessing Pipeline

import numpy as np

import Utils.angle_utils as angle_utils
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def calculate_angular_distance_for_gaze_data(trial: LWSTrial, sm: ScreenMonitor) -> np.ndarray:
    """
    Calculates the angular distance between each gaze-point in the trial, and the nearest trial's target.
        Returns np.inf for gaze-points with missing or invalid data.
    """
    _, xs, ys = trial.get_raw_gaze_coordinates(eye='dominant')
    target_data = trial.get_stimulus().get_target_data()
    d = trial.get_subject_info().distance_to_screen

    angular_distances = np.ones_like(xs) * np.inf
    for i, row in target_data.iterrows():
        tx, ty = row['center_x'], row['center_y']
        dist = calculate_angular_distance_to_target(tx, ty, xs, ys, d, sm)
        angular_distances = np.nanmin([angular_distances, dist], axis=0)
    return angular_distances


def calculate_angular_target_distance_for_fixation(fix: LWSFixationEvent,
                                                   trial: LWSTrial,
                                                   sm: ScreenMonitor) -> float:
    """
    Calculates the angular distance between the fixation event's center-of-mass, and the nearest trial's target.
        Returns np.inf if the fixation event's center-of-mass is missing or invalid.
    """
    x, y = fix.center_of_mass
    target_data = trial.get_stimulus().get_target_data()
    d = trial.get_subject_info().distance_to_screen

    angular_distance = np.inf
    for i, row in target_data.iterrows():
        tx, ty = row['center_x'], row['center_y']
        dist = calculate_angular_distance_to_target(tx, ty, np.array([x]), np.array([y]), d, sm)
        angular_distance = np.nanmin([angular_distance, dist[0]])
    return angular_distance


def calculate_angular_distance_to_target(tx: float, ty: float, xs: np.ndarray, ys: np.ndarray,
                                         d: float, sm: ScreenMonitor) -> np.ndarray:
    """
    Calculate the angular distance (in degrees) between the target and each of the gaze points.
    :param tx, ty: target x, y coordinates
    :param xs, ys: arrays of gaze x, y coordinates
    :param d: distance of the viewer from the screen
    :param sm: ScreenMonitor object
    """
    if len(xs) != len(ys):
        raise ValueError("x and y must be of the same length")
    if np.isnan(tx) or np.isnan(ty):
        raise ValueError("Target coordinates cannot be NaN")
    if np.isnan(d):
        raise ValueError("Distance cannot be NaN")

    distances = np.zeros_like(xs)
    for i in range(len(xs)):
        distances[i] = angle_utils.calculate_visual_angle(d=d, p1=(tx, ty), p2=(xs[i], ys[i]),
                                                          screen_monitor=sm, use_radians=False)
    return distances


def calculate_euclidean_distance_to_target(tx: float, ty: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Returns the euclidean distance (in pixels) between the target and each of the gaze points.
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
