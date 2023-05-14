import numpy as np

import experiment_config as cnfg
from Utils.ScreenMonitor import ScreenMonitor
from LWS.DataModels.LWSTrial import LWSTrial


def check_target_proximity_for_gaze_data(trial: LWSTrial, sm: ScreenMonitor,
                                         angle: float = cnfg.THRESHOLD_VISUAL_ANGLE) -> np.ndarray:
    """
    For all gaze points in the trial, check if they are within a certain visual angle from any of the trial's targets.
    Returns a boolean array of the same length as the number of gaze points, where True means that the gaze point is
        within the visual angle threshold from at least one target.
    """
    _, xs, ys = trial.get_raw_gaze_coordinates()
    target_data = trial.stimulus.get_target_data()
    d = trial.subject_info.distance_to_screen

    is_close_to_any_target = np.zeros_like(xs, dtype=bool)
    for i, row in enumerate(target_data):
        tx, ty = row['center_x'], row['center_y']
        is_close_to_target = check_proximity_to_target(tx, ty, xs, ys, d, angle, sm)
        is_close_to_any_target = np.logical_or(is_close_to_any_target, is_close_to_target)
    return is_close_to_any_target


def check_proximity_to_target(tx: float, ty: float,
                              xs: np.ndarray, ys: np.ndarray,
                              d: float, angle: float,
                              sm: ScreenMonitor) -> np.ndarray:
    """
    Check if the gaze point is within a certain distance from the target.
    :param tx, ty: target x, y coordinates
    :param xs, ys: arrays of gaze x, y coordinates
    :param d: distance of the viewer from the screen
    :param angle: visual angle considered proximal to the target
    :param sm: ScreenMonitor object

    Returns a boolean array of the same length as x and y, where True indicates that the gaze point is within a visual
        angle of `angle` from the target.

    :raises ValueError: if x and y are not of the same length, if any of the given coordinates is NaN, or if d or angle
        are NaN.
    """
    if len(xs) != len(ys):
        raise ValueError("x and y must be of the same length")
    if np.isnan(tx) or np.isnan(ty):
        raise ValueError("Target coordinates cannot be NaN")
    if np.isnan(d) or np.isnan(angle):
        raise ValueError("Distance and angle cannot be NaN")

    is_close = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        angle_between = sm.calc_angle_between_pixels(d, p1=(tx, ty), p2=(xs[i], ys[i]), use_radians=False)
        if np.isnan(angle_between):
            is_close[i] = False
        else:
            is_close[i] = angle_between <= angle
    return is_close
