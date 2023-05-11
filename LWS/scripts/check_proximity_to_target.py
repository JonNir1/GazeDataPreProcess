import numpy as np

from Utils.ScreenMonitor import ScreenMonitor


def check_proximity_to_target(tx: float, ty: float,
                              x: np.ndarray, y: np.ndarray,
                              d: float, angle: float,
                              sm: ScreenMonitor) -> np.ndarray:
    """
    Check if the gaze point is within a certain distance from the target.
    :param tx, ty: target x, y coordinates
    :param x, y: arrays of gaze x, y coordinates
    :param d: distance of the viewer from the screen
    :param angle: visual angle considered proximal to the target
    :param sm: ScreenMonitor object

    Returns a boolean array of the same length as x and y, where True indicates that the gaze point is within a visual
        angle of `angle` from the target.

    :raises ValueError: if x and y are not of the same length, if any of the given coordinates is NaN, or if d or angle
        are NaN.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length")
    if np.isnan(tx) or np.isnan(ty):
        raise ValueError("Target coordinates cannot be NaN")
    if np.isnan(d) or np.isnan(angle):
        raise ValueError("Distance and angle cannot be NaN")

    is_close = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        angle_between = sm.calc_angle_between_pixels(d, p1=(tx, ty), p2=(x[i], y[i]), use_radians=False)
        if np.isnan(angle_between):
            is_close[i] = False
        else:
            is_close[i] = angle_between <= angle
    return is_close
