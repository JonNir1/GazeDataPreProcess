import numpy as np
import pandas as pd
from typing import List, Dict

import constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from LWS.DataModels.LWSSaccadeEvent import LWSSaccadeEvent
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

    event_types = np.unique([e.event_type().lower() for e in events])
    if len(event_types) != 1:
        # summary of multiple event types
        data = __get_basic_summary_dict(events=events)
        return pd.Series(data)

    # summary of single event type
    event_type = event_types[0]
    if event_type == cnst.BLINK:
        data = __get_basic_summary_dict(events=events)
        return pd.Series(data)

    if event_type == cnst.SACCADE:
        events: List[LWSSaccadeEvent]
        data = __get_saccade_summary_dict(saccades=events)
        return pd.Series(data)

    if event_type == cnst.FIXATION:
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


def __get_saccade_summary_dict(saccades: List[LWSSaccadeEvent]) -> Dict[str, float]:
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
    data["visual_angle_mean"] = np.nanmean([s.visual_angle for s in saccades])
    data["visual_angle_std"] = np.nanstd([s.visual_angle for s in saccades])
    return data


def __get_fixation_summary_dict(fixations: List[LWSFixationEvent]) -> Dict[str, float]:
    """
    Extracts a dictionary containing summary of fixations, with the following keys:
        - count
        - duration_mean
        - duration_std
        - x_dispersion_mean
        - x_dispersion_std
        - y_dispersion_mean
        - y_dispersion_std
        - distance_to_target_mean
        - distance_to_target_std

    :param fixations: list of fixations to summarize

    :return: dictionary containing the summary of the fixations (with or without outliers)
    """
    data = __get_basic_summary_dict(events=fixations)
    data["x_dispersion_mean"] = np.nanmean([f.std[0] for f in fixations])
    data["x_dispersion_std"] = np.nanstd([f.std[0] for f in fixations])
    data["y_dispersion_mean"] = np.nanmean([f.std[1] for f in fixations])
    data["y_dispersion_std"] = np.nanstd([f.std[1] for f in fixations])
    data["visual_angle_to_target_mean"] = np.nanmean(
        [f.distance_to_target for f in fixations if np.isfinite(f.visual_angle_to_target)])
    data["visual_angle_to_target_std"] = np.nanstd(
        [f.distance_to_target for f in fixations if np.isfinite(f.visual_angle_to_target)])
    return data
