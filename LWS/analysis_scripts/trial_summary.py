import numpy as np
import pandas as pd
import warnings as w  # to suppress numpy warnings
from typing import List

import constants as cnst
from Config import experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


def summarize_all_trials(trials: List[LWSTrial], safe=True) -> pd.DataFrame:
    """
    Extracts a summary of events during each trial, containing the following information:
        - trial (trial number, int) - index
        - trial duration
        - number of events of each type
        - number of outlier events of each type
        - mean duration of each event type (with and without outliers)
        - mean distance to target for fixation events (with and without outliers)
        - mean visual angle (amplitude) for saccade events (with and without outliers)
        - total event count
        - total outlier event count

    :returns: pandas DataFrame containing the summary of all trials
    """
    series_list = []
    for tr in trials:
        try:
            s = summarize_single_trial(tr, suppress_warnings=safe)
            series_list.append(s)
        except Exception as e:
            if not safe:
                raise e
    df = pd.DataFrame(series_list).set_index(cnst.TRIAL)
    df.index = df.index.astype(int)  # convert trial number to int
    return df


def summarize_single_trial(trial: LWSTrial, suppress_warnings=True) -> pd.Series:
    """
    Extracts a summary of events during the trial, containing the following information:
        - trial (trial number)
        - trial duration
        - number of events of each type
        - number of outlier events of each type
        - mean duration of each event type (with and without outliers)
        - mean distance to target for fixation events (with and without outliers)
        - mean visual angle (amplitude) for saccade events (with and without outliers)
        - total event count
        - total outlier event count

    If `suppress_warnings` is True, numpy warnings will be suppressed.
    :returns: pandas Series containing the summary of the trial
    """
    if suppress_warnings:
        with w.catch_warnings():
            w.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            return __summarize_single_trial_unsafe(trial)
    return __summarize_single_trial_unsafe(trial)


def __summarize_single_trial_unsafe(trial: LWSTrial) -> pd.Series:
    if not trial.is_processed:
        raise RuntimeError(f"Trial {trial} is not processed")
    trial_data = {cnst.TRIAL: trial.trial_num, 'duration': trial.duration}
    for et in GazeEventTypeEnum:
        events: List[BaseGazeEvent] = trial.get_gaze_events(event_type=et)
        events_count = len(events)
        events_outlier_count = len([e for e in events if e.is_outlier])
        trial_data[f"{et}_count"] = events_count
        trial_data[f"{et}_outlier_count"] = events_outlier_count
        trial_data[f"{et}_mean_duration"] = np.nanmean([e.duration for e in events])
        trial_data[f"{et}_mean_duration_no_outliers"] = np.nanmean([e.duration for e in events if not e.is_outlier])

        if et == cnst.FIXATION:
            from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
            events: List[LWSFixationEvent]
            trial_data[f"{et}_mean_distance_to_target"] = np.nanmean([e.visual_angle_to_target for e in events if
                                                                      np.isfinite(e.visual_angle_to_target)])
            trial_data[f"{et}_mean_distance_to_target_no_outliers"] = np.nanmean(
                [e.visual_angle_to_target for e in events if
                 np.isfinite(e.visual_angle_to_target) and not e.is_outlier])
        elif et == cnst.SACCADE:
            from GazeEvents.SaccadeEvent import SaccadeEvent
            events: List[SaccadeEvent]
            trial_data[f"{et}_mean_visual_angle"] = np.nanmean(
                [e.amplitude for e in events if np.isfinite(e.amplitude)])
            trial_data[f"{et}_mean_visual_angle_no_outliers"] = np.nanmean([e.amplitude for e in events if
                                                                            np.isfinite(
                                                                                e.amplitude) and not e.is_outlier])

    trial_data["total_events_count"] = sum([trial_data[f"{et}_count"] for et in cnfg.EVENT_TYPES])
    trial_data["total_outliers_count"] = sum([trial_data[f"{et}_outlier_count"] for et in cnfg.EVENT_TYPES])
    return pd.Series(trial_data)
