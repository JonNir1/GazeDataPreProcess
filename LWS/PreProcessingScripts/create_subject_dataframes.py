# LWS PreProcessing Pipeline

import time

import Utils.io_utils as ioutils
from LWS.DataModels.LWSSubject import LWSSubject


def create_subject_dataframes(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    _subject_dataframes_dir = ioutils.create_directory(dirname="dataframes", parent_dir=subject.output_dir)
    trial_summary_df = _trial_summary(subject, save)
    trigger_counts = _trigger_summary(subject, save)
    lws_instances = _lws_identification(subject, save)
    lws_rates_all_fixations, lws_rates_proximal_fixations = _lws_rate(subject, save)
    r2roi_counts_exclude_rect, r2roi_counts_include_rect = _return_to_roi(subject, save)
    end = time.time()
    if verbose:
        ioutils.print_and_log(msg="Finished creating DataFrames for subject " +
                                  f"{subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return (trial_summary_df, trigger_counts, lws_instances, lws_rates_all_fixations, lws_rates_proximal_fixations,
            r2roi_counts_exclude_rect, r2roi_counts_include_rect)


def _trial_summary(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.event_analysis.trial_summary as trsum
    trial_summary_df = trsum.summarize_all_trials(subject.get_trials())
    subject.set_dataframe(trsum.DF_NAME, trial_summary_df)
    if save:
        trial_summary_df.to_pickle(subject.get_dataframe_path(trsum.DF_NAME))
    return trial_summary_df


def _trigger_summary(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.event_analysis.triggers_counts as trig
    trigger_counts = trig.count_triggers_per_trial(subject)
    subject.set_dataframe(trig.DF_NAME, trigger_counts)
    if save:
        trigger_counts.to_pickle(subject.get_dataframe_path(trig.DF_NAME))
    return trigger_counts


def _lws_identification(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.identify_lws_instances as lws_inst
    lws_instances = lws_inst.identify_lws_for_varying_thresholds(subject)
    subject.set_dataframe(lws_inst.INSTANCES_DF_NAME, lws_instances)
    if save:
        lws_instances.to_pickle(subject.get_dataframe_path(lws_inst.INSTANCES_DF_NAME))
    return lws_instances


def _lws_rate(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.identify_lws_instances as lws_inst

    # calculate LWS rates out of all fixations:
    all_fixs_df_name = lws_inst.RATES_DF_BASE_NAME + "_all_fixations"
    lws_rates_all_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=False)
    subject.set_dataframe(all_fixs_df_name, lws_rates_all_fixations)

    # calculate LWS rates out of target-proximal fixations:
    prox_fixs_df_name = lws_inst.RATES_DF_BASE_NAME + "_proximal_fixations"
    lws_rates_proximal_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=True)
    subject.set_dataframe(prox_fixs_df_name, lws_rates_proximal_fixations)

    if save:
        lws_rates_all_fixations.to_pickle(subject.get_dataframe_path(all_fixs_df_name))
        lws_rates_proximal_fixations.to_pickle(subject.get_dataframe_path(prox_fixs_df_name))
    return lws_rates_all_fixations, lws_rates_proximal_fixations


def _return_to_roi(subject: LWSSubject, save: bool):
    import LWS.SubjectAnalysis.search_analysis.return_to_roi as r2roi

    # calculate return-to-ROI counts when the bottom rectangle is not part of the ROI:
    exclude_rect_df_name = r2roi.BASE_DF_NAME + "_exclude_rect"
    r2roi_counts_exclude_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                is_targets_rect_part_of_roi=False)
    subject.set_dataframe(exclude_rect_df_name, r2roi_counts_exclude_rect)

    # calculate return-to-ROI counts when the bottom rectangle is part of the ROI:
    include_rect_df_name = r2roi.BASE_DF_NAME + "_include_rect"
    r2roi_counts_include_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                is_targets_rect_part_of_roi=True)
    subject.set_dataframe(include_rect_df_name, r2roi_counts_include_rect)

    if save:
        r2roi_counts_exclude_rect.to_pickle(subject.get_dataframe_path(exclude_rect_df_name))
        r2roi_counts_include_rect.to_pickle(subject.get_dataframe_path(include_rect_df_name))
    return r2roi_counts_exclude_rect, r2roi_counts_include_rect

