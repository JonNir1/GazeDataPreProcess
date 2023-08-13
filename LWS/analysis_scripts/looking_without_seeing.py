import numpy as np

import Config.experiment_config as cnfg
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def count_lws_fixations_per_trial(trial: LWSTrial,
                                  ignore_outliers: bool = True,
                                  proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE) -> int:
    """
    Counts the number of Looking-Without-Seeing events in the trial.
    A LWS event is a fixation that is close to a target that was not identified up until the end of the fixation.
    """
    counter = 0
    target_info = trial.get_targets(proximity_threshold=proximity_threshold)
    fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION, ignore_outliers=ignore_outliers)
    for f in fixations:
        f: LWSFixationEvent
        if f.visual_angle_to_closest_target > proximity_threshold:
            # fixation is not close to any target
            continue

        closest_target_id = f.closest_target - 1
        time_identified = target_info.loc[closest_target_id, "time_identified"]
        if np.isnan(time_identified):
            # fixation is close to a never-identified target --> increase counter
            counter += 1
            continue

        if f.start_time > time_identified:
            # fixation is close to a target that was identified before the fixation started
            # this is a target-return fixation, not a looking-without-seeing fixation
            continue

        if f.end_time < time_identified:
            # fixation is close to a target that was identified after the fixation ended
            # TODO: check if the next fixation is close to the same target and is also before the target was identified
            # TODO: check if the next fixation is within the target-helper region of the stimulus
            # TODO: check if the next fixation is within the image-array region of the stimulus -- only here increase counter
            counter += 1
            continue
    return counter

