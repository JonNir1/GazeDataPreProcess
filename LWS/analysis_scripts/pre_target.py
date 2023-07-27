import numpy as np
from typing import List, Tuple

import Config.experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


# TODO: compare saccades leading to target-marking vs. target-proximal fixations


def __get_target_fixations_previous_events(trials: List[LWSTrial],
                                           proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE
                                           ) -> Tuple[List[BaseGazeEvent], List[BaseGazeEvent]]:
    """
    Returns a list of all events in the given trials, that preceded a target-proximal fixation event, and a list of all
    events that preceded a target-marking fixation event.
    """
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    proximal_fixation_previous_events = []
    marking_fixation_previous_events = []
    for trial in trials:
        proximal_fixation_previous_events_trial, marking_fixation_previous_events_trial = \
            __get_target_fixations_previous_events_single_trial(trial, proximity_threshold)
        proximal_fixation_previous_events.extend(proximal_fixation_previous_events_trial)
        marking_fixation_previous_events.extend(marking_fixation_previous_events_trial)
    return proximal_fixation_previous_events, marking_fixation_previous_events


def __get_target_fixations_previous_events_single_trial(trial: LWSTrial,
                                                        proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE
                                                        ) -> Tuple[List[BaseGazeEvent], List[BaseGazeEvent]]:
    """
    Returns a list of all events in the given trial, that preceded a target-proximal fixation event, and a list of all
    events that preceded a target-marking fixation event.
    """
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        raise ValueError(f"Invalid proximity threshold: {proximity_threshold}")
    events = trial.get_gaze_events()
    proximal_fixation_previous_events = []
    marking_fixation_previous_events = []
    for i, e in enumerate(events):
        if i == len(events) - 1:
            continue
        if events[i + 1].event_type() != GazeEventTypeEnum.FIXATION:
            continue
        next_event: LWSFixationEvent = events[i + 1]
        if next_event.visual_angle_to_target <= cnfg.THRESHOLD_VISUAL_ANGLE:
            proximal_fixation_previous_events.append(e)
        if next_event.is_mark_target_attempt:
            marking_fixation_previous_events.append(e)
    return proximal_fixation_previous_events, marking_fixation_previous_events

