# LWS PreProcessing Pipeline

import time
import numpy as np

import Config.experiment_config as cnfg
import Utils.io_utils as ioutils
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.DataModels.LWSSubject import LWSSubject


_PROX_THRESHOLDS = np.arange(cnfg.THRESHOLD_VISUAL_ANGLE / 15,
                             21 * cnfg.THRESHOLD_VISUAL_ANGLE / 15,
                             cnfg.THRESHOLD_VISUAL_ANGLE / 15)
_TIME_DIFF_THRESHOLDS = np.arange(0, SaccadeEvent.MAX_DURATION + 1, 50)


def create_subject_dataframes(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    _subject_dataframes_dir = ioutils.create_directory(dirname="dataframes", parent_dir=subject.output_dir)
    _trial_summary(subject, save)
    _trigger_summary(subject, save)
    _lws_identification(subject, save)
    _lws_rate(subject, save)
    _return_to_roi(subject, save)
    end = time.time()
    if verbose:
        ioutils.log_and_print(msg="Finished creating DataFrames for subject " +
                                  f"{subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)


def _trial_summary(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.event_analysis.trial_summary as trsum
    trial_summary_df = subject.get_dataframe(trsum.DF_NAME)
    if trial_summary_df is None:
        trial_summary_df = trsum.summarize_all_trials(subject.get_trials())
        subject.set_dataframe(trsum.DF_NAME, trial_summary_df)

    if save:
        trial_summary_df.to_pickle(subject.get_dataframe_path(trsum.DF_NAME))
    return trial_summary_df


def _trigger_summary(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.event_analysis.triggers_counts as trig
    trigger_counts = subject.get_dataframe(trig.DF_NAME)
    if trigger_counts is None:
        trigger_counts = trig.count_triggers_per_trial(subject)
        subject.set_dataframe(trig.DF_NAME, trigger_counts)

    if save:
        trigger_counts.to_pickle(subject.get_dataframe_path(trig.DF_NAME))
    return trigger_counts


def _lws_identification(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.identify_lws_instances as lws_inst
    lws_instances = subject.get_dataframe(lws_inst.INSTANCES_DF_NAME)
    if lws_instances is None:
        lws_instances = lws_inst.identify_lws_for_varying_thresholds(subject,
                                                                     proximity_thresholds=_PROX_THRESHOLDS,
                                                                     time_difference_thresholds=_TIME_DIFF_THRESHOLDS)
        subject.set_dataframe(lws_inst.INSTANCES_DF_NAME, lws_instances)

    if save:
        lws_instances.to_pickle(subject.get_dataframe_path(lws_inst.INSTANCES_DF_NAME))
    return lws_instances


def _lws_rate(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.identify_lws_instances as lws_inst

    # calculate LWS rates out of all fixations:
    df_name = lws_inst.RATES_DF_BASE_NAME + "_all_fixations"
    lws_rates_all_fixations = subject.get_dataframe(df_name)
    if lws_rates_all_fixations is None:
        lws_rates_all_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=False)
        subject.set_dataframe(df_name, lws_rates_all_fixations)

    # calculate LWS rates out of target-proximal fixations:
    df_name = lws_inst.RATES_DF_BASE_NAME + "_proximal_fixations"
    lws_rates_proximal_fixations = subject.get_dataframe(df_name)
    if lws_rates_proximal_fixations is None:
        lws_rates_proximal_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=True)
        subject.set_dataframe(df_name, lws_rates_proximal_fixations)

    if save:
        lws_rates_all_fixations.to_pickle(subject.get_dataframe_path(lws_rates_all_fixations.name))
        lws_rates_proximal_fixations.to_pickle(subject.get_dataframe_path(lws_rates_proximal_fixations.name))
    return lws_rates_all_fixations, lws_rates_proximal_fixations


def _return_to_roi(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.return_to_roi as r2roi

    # calculate return-to-ROI counts when the bottom rectangle is not part of the ROI:
    df_name = r2roi.BASE_DF_NAME + "_exclude_rect"
    r2roi_counts_exclude_rect = subject.get_dataframe(df_name)
    if r2roi_counts_exclude_rect is None:
        r2roi_counts_exclude_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                proximity_thresholds=_PROX_THRESHOLDS,
                                                                                                is_targets_rect_part_of_roi=False)
        subject.set_dataframe(df_name, r2roi_counts_exclude_rect)

    # calculate return-to-ROI counts when the bottom rectangle is part of the ROI:
    df_name = r2roi.BASE_DF_NAME + "_include_rect"
    r2roi_counts_include_rect = subject.get_dataframe(df_name)
    if r2roi_counts_include_rect is None:
        r2roi_counts_include_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                proximity_thresholds=_PROX_THRESHOLDS,
                                                                                                is_targets_rect_part_of_roi=True)
        subject.set_dataframe(df_name, r2roi_counts_include_rect)

    if save:
        r2roi_counts_exclude_rect.to_pickle(subject.get_dataframe_path(r2roi.BASE_DF_NAME + "_exclude_rect"))
        r2roi_counts_include_rect.to_pickle(subject.get_dataframe_path(r2roi.BASE_DF_NAME + "_include_rect"))
    return r2roi_counts_exclude_rect, r2roi_counts_include_rect

