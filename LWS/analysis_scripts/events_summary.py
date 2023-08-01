import numpy as np
import pandas as pd
from typing import List, Dict

import constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.BlinkEvent import BlinkEvent
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def summarize_events(events: List[BaseGazeEvent], filter_outliers=False) -> pd.Series:
    """
    Extracts a pd.Series containing summary of events, with the following keys:
        - count
        - duration_mean
        - duration_std

        for saccade events:
        - visual_angle_mean
        - visual_angle_std

        for fixation events:
        - x_dispersion_mean
        - x_dispersion_std
        - y_dispersion_mean
        - y_dispersion_std
        - distance_to_target_mean
        - distance_to_target_std

    :param events: list of events to summarize
    :param filter_outliers: if True, outliers will be filtered out before calculating summary statistics

    :return: pd.Series containing summary of events
    """
    if filter_outliers:
        events: List[BaseGazeEvent] = [e for e in events if not e.is_outlier]

    event_types = set([e.event_type() for e in events])
    if len(event_types) != 1:
        # summary of multiple event types
        data = __get_basic_summary_dict(events=events)
        return pd.Series(data)

    # summary of single event type
    event_type = event_types.pop()
    if event_type == BlinkEvent.event_type():
        data = __get_basic_summary_dict(events=events)
        return pd.Series(data)

    if event_type == SaccadeEvent.event_type():
        events: List[SaccadeEvent]
        data = __get_saccade_summary_dict(saccades=events)
        return pd.Series(data)

    if event_type == LWSFixationEvent.event_type():
        events: List[LWSFixationEvent]
        data = __get_fixation_summary_dict(fixations=events)
        return pd.Series(data)

    raise ValueError(f"Unknown event type: {event_type}")


def __get_basic_summary_dict(events: List[BaseGazeEvent]) -> Dict[str, float]:
    """
    Extracts a dictionary containing basic summary of events, with the following keys:
        - count
        - duration_mean
        - duration_std

    :param events: list of events to summarize

    :return: dictionary containing basic summary of events
    """
    data = {
        "count": len(events),
        "duration_mean": np.nanmean([e.duration for e in events]),
        "duration_std": np.nanstd([e.duration for e in events]),
    }
    return data


def __get_saccade_summary_dict(saccades: List[SaccadeEvent]) -> Dict[str, float]:
    """
    Extracts a dictionary containing summary of saccades, with the following keys:
        - count
        - duration_mean
        - duration_std
        - visual_angle_mean
        - visual_angle_std

    :param saccades: list of saccades to summarize

    :return: dictionary containing summary of saccades
    """
    data = __get_basic_summary_dict(events=saccades)
    data["visual_angle_mean"] = np.nanmean([s.amplitude for s in saccades])
    data["visual_angle_std"] = np.nanstd([s.amplitude for s in saccades])
    return data


def __get_fixation_summary_dict(fixations: List[LWSFixationEvent]) -> Dict[str, float]:
    """
    Extracts a dictionary containing summary of fixations, with the following keys:
        - count
        - duration_mean
        - duration_std
        - x_std_mean
        - y_std_mean
        - mean_visual_angle_to_closest_target
        - std_visual_angle_to_closest_target

    :param fixations: list of fixations to summarize

    :return: dictionary containing the summary of the fixations (with or without outliers)
    """
    data = __get_basic_summary_dict(events=fixations)
    stds = [f.standard_deviation for f in fixations]
    data["x_std_mean"] = float(np.nanmean([std[0] for std in stds]))
    data["y_std_mean"] = float(np.nanmean([std[1] for std in stds]))
    data["pupil_size_mean"] = float(np.nanmean([f.mean_pupil_size for f in fixations]))
    data["pupil_size_std"] = float(np.nanstd([f.mean_pupil_size for f in fixations]))

    # calculate the visual angle to closest target for each fixation, and discard NaNs and Inf
    vis_angle_to_closest_target = [min(f.visual_angle_to_targets) for f in fixations]
    vis_angle_to_closest_target = list(filter(lambda x: np.isfinite(x), vis_angle_to_closest_target))

    # these may warn `RuntimeWarning: Mean of empty slice`
    data["mean_visual_angle_to_closest_target"] = float(np.nanmean(vis_angle_to_closest_target))
    data["std_visual_angle_to_closest_target"] = float(np.nanstd(vis_angle_to_closest_target))
    return data
