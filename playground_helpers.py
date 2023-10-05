import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import Config.experiment_config as cnfg
import Utils.io_utils as ioutils
import Visualization.visualization_utils as visutils
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def full_pipline(name: str,
                 save: bool = True,
                 include_subject_dfs: bool = True,
                 include_subject_figures: bool = True,
                 include_trial_figures: bool = True,
                 include_trial_videos: bool = True,
                 verbose: bool = True):
    start = time.time()
    print(f"Processing subject `{name}`...")
    subject = process_subject(name=name, save=save, verbose=verbose)
    subject = load_subject(subject_id=subject.subject_id, verbose=verbose)

    subject_dfs = None
    if include_subject_dfs:
        subject_dfs = create_subject_dataframes(subject=subject, save=save, verbose=verbose)

    subject_figures = None
    if include_subject_figures:
        subject_figures = create_subject_figures(subject=subject, save=save, verbose=verbose)

    failed_analysis_trials = []
    if include_trial_figures:
        failed_analysis_trials = analyze_all_trials(subject=subject, save=save, verbose=verbose)

    failed_video_trials = []
    if include_trial_videos:
        failed_video_trials = create_trial_videos(subject=subject, save=save, verbose=verbose)

    failed_trials = failed_analysis_trials + failed_video_trials
    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"\nFinished processing subject {name}: {(end - start):.2f} seconds\n###############\n",
                              log_file=subject.log_file)
    return subject, subject_dfs, subject_figures, failed_trials


def process_subject(name: str, save: bool = False, verbose: bool = True) -> LWSSubject:
    start = time.time()
    import LWS.PreProcessing as pp
    subject = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, name),
                                 screen_monitor=cnfg.SCREEN_MONITOR,
                                 save_pickle=save,
                                 stuff_with='fixation',
                                 blink_detector_type='missing data',
                                 saccade_detector_type='engbert',
                                 drop_outlier_events=False)
    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"Finished preprocessing subject `{name}`: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return subject


def load_subject(subject_id: int, verbose: bool = True) -> LWSSubject:
    start = time.time()
    subject_id = f"{subject_id:03d}"
    subdir = f"S{subject_id}"
    subject = LWSSubject.from_pickle(os.path.join(cnfg.OUTPUT_DIR, subdir, f"LWSSubject_{subject_id}.pkl"))
    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"Finished loading subject {subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return subject


def create_subject_dataframes(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    _subject_dataframes_dir = ioutils.create_directory(dirname="dataframes", parent_dir=subject.output_dir)

    import LWS.Analysis.event_analysis.trial_summary as trsum
    trial_summary = trsum.summarize_all_trials(subject.get_trials())
    if save:
        trial_summary.to_pickle(subject.get_dataframe_path(trsum.DF_NAME))

    import LWS.Analysis.event_analysis.triggers_analysis as trig
    trigger_counts = trig.count_triggers_per_trial(subject)
    if save:
        trigger_counts.to_pickle(subject.get_dataframe_path(trig.DF_NAME))

    import LWS.Analysis.search_analysis.identify_lws_instances as lws_inst
    lws_instances = lws_inst.identify_lws_for_varying_thresholds(subject,
                                                                 proximity_thresholds=np.arange(0.1, 7.1, 0.1),
                                                                 time_difference_thresholds=np.arange(0, 251, 10))
    if save:
        lws_instances.to_pickle(subject.get_dataframe_path(lws_inst.INSTANCES_DF_NAME))

    lws_rates_all_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=False)
    if save:
        lws_rates_all_fixations.to_pickle(subject.get_dataframe_path(lws_inst.RATES_DF_BASE_NAME + "_all_fixations"))

    lws_rates_proximal_fixations = lws_inst.calculate_lws_rates(subject, proximal_fixations_only=True)
    if save:
        lws_rates_proximal_fixations.to_pickle(subject.get_dataframe_path(lws_inst.RATES_DF_BASE_NAME + "_proximal_fixations"))

    import LWS.Analysis.search_analysis.return_to_roi as r2roi
    r2roi_counts_exclude_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                proximity_thresholds=np.arange(
                                                                                                    0.1, 7.1, 0.1),
                                                                                                is_targets_rect_part_of_roi=False)
    r2roi_counts_include_rect = r2roi.count_fixations_between_roi_visits_for_varying_thresholds(subject,
                                                                                                proximity_thresholds=np.arange(
                                                                                                    0.1, 7.1, 0.1),
                                                                                                is_targets_rect_part_of_roi=True)
    if save:
        r2roi_counts_exclude_rect.to_pickle(subject.get_dataframe_path(r2roi.DF_NAME + "_exclude_rect"))
        r2roi_counts_include_rect.to_pickle(subject.get_dataframe_path(r2roi.DF_NAME + "_include_rect"))

    end = time.time()
    if verbose:
        ioutils.log_and_print(msg="Finished creating DataFrames for subject " +
                                  f"{subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return trial_summary, trigger_counts, lws_instances


def create_subject_figures(subject: LWSSubject, proximity_threshold: float = cnfg.THRESHOLD_VISUAL_ANGLE,
                           save: bool = False, verbose: bool = True):
    start = time.time()

    trials = subject.get_trials()
    all_saccades: List[SaccadeEvent] = [s for tr in trials for s in tr.get_gaze_events(GazeEventTypeEnum.SACCADE)]
    all_fixations: List[LWSFixationEvent] = [f for tr in trials for f in tr.get_gaze_events(GazeEventTypeEnum.FIXATION)]
    subject_figures_dir = ioutils.create_directory(dirname="subject_figures", parent_dir=subject.output_dir)

    import Visualization.saccade_analysis as sacan
    saccade_distributions = sacan.distributions_figure(all_saccades, ignore_outliers=True,
                                                       title="Saccades Property Distributions", show_legend=True)
    if save:
        visutils.save_figure(saccade_distributions,
                             full_path=os.path.join(subject_figures_dir, "saccade distributions.png"))

    import LWS.Analysis.event_analysis.fixation_analysis as fixan
    target_proximal_fixations, target_marking_fixations, target_distal_fixations = fixan.split_by_target_proximity(
        all_fixations, proximity_threshold)
    fixation_groups = [all_fixations, target_distal_fixations, target_proximal_fixations, target_marking_fixations]
    group_names = ["All Fixations", "Distal Fixations", "Proximal Fixations", "Marking Fixations"]

    all_distribution_comparison = fixan.plot_feature_distributions(fixation_groups, group_names,
                                                                   title="All Fixation Types",
                                                                   show_legend=True)
    if save:
        visutils.save_figure(all_distribution_comparison,
                             full_path=os.path.join(subject_figures_dir, "feature distribution - all_fixations.png"))

    proximal_distribution_comparison = fixan.plot_feature_distributions(fixation_groups[2:], group_names[2:],
                                                                        title="Proximal (Non-Marking) vs. Marking Fixations",
                                                                        show_legend=True)
    if save:
        visutils.save_figure(proximal_distribution_comparison,
                             full_path=os.path.join(subject_figures_dir, "feature distribution - proximal_fixations.png"))

    distal_distribution_comparison = fixan.plot_feature_distributions(fixation_groups[1:3], group_names[1:3],
                                                                      title="Distal vs. Proximal (Non-Marking) Fixations",
                                                                      show_legend=True)
    if save:
        visutils.save_figure(distal_distribution_comparison,
                             full_path=os.path.join(subject_figures_dir, "feature distribution - distal_fixations.png"))

    # plot_feature_dynamics
    fixation_dynamics = fixan.plot_feature_dynamics(fixation_groups, group_names,
                                                    title="Fixations Temporal Dynamics", show_legend=True)
    if save:
        visutils.save_figure(fixation_dynamics,
                             full_path=os.path.join(subject_figures_dir, "fixation dynamics - all_fixations.png"))

    import LWS.Analysis.search_analysis.lws_instance_rate as lws_rate
    lws_rates_fig = lws_rate.lws_rates_figure(subject,
                                              proximity_thresholds=np.arange(cnfg.THRESHOLD_VISUAL_ANGLE / 15,
                                                                             21 * cnfg.THRESHOLD_VISUAL_ANGLE / 15,
                                                                             cnfg.THRESHOLD_VISUAL_ANGLE / 15),
                                              time_difference_thresholds=np.arange(0,
                                                                                   SaccadeEvent.MAX_DURATION + 1,
                                                                                   50))
    if save:
        visutils.save_figure(lws_rates_fig,
                             full_path=os.path.join(subject_figures_dir, "lws rates.png"))

    import LWS.Analysis.event_analysis.triggers_analysis as trig
    trigger_rates = trig.plot_trigger_rates_by_block_position(subject)
    if save:
        visutils.save_figure(trigger_rates,
                             full_path=os.path.join(subject_figures_dir, "trigger rates.png"))

    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"Finished analyzing subject {subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return (saccade_distributions, all_distribution_comparison, proximal_distribution_comparison,
            distal_distribution_comparison, fixation_dynamics, lws_rates_fig, trigger_rates)


def analyze_all_trials(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    from LWS.TrialVisualizer.LWSTrialGazeVisualizer import LWSTrialGazeVisualizer
    from LWS.TrialVisualizer.LWSTrialTargetDistancesVisualizer import LWSTrialTargetDistancesVisualizer
    from LWS.TrialVisualizer.LWSTrialHeatmapVisualizer import LWSTrialGazeHeatmapVisualizer, LWSTrialFixationsHeatmapVisualizer

    failed_trials = []
    for tr in subject.get_trials():
        try:
            start_trial = time.time()
            _gaze = LWSTrialGazeVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                           output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            _targets = LWSTrialTargetDistancesVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                         output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            _gaze_heatmap = LWSTrialGazeHeatmapVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                          output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            _fixation_heatmap = LWSTrialFixationsHeatmapVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                                   output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            plt.close('all')  # close all open figures from memory
            end_trial = time.time()
            if verbose:
                ioutils.log_and_print(
                    msg=f"\t{tr.__repr__()} Analysis:\t{(end_trial - start_trial):.2f} s", log_file=subject.log_file)
        except Exception as _e:
            trace = traceback.format_exc()
            failed_trials.append((tr, trace))
            if verbose:
                ioutils.log_and_print(
                    msg=f"######\n\tFailed to analyze trial {tr.__repr__()}:\n\t{trace}\n", log_file=subject.log_file)

    end = time.time()
    if verbose:
        ioutils.log_and_print(
            msg=f"Finished analyzing all trials: {(end - start):.2f} seconds", log_file=subject.log_file)
    return failed_trials


def create_trial_videos(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    from LWS.TrialVisualizer.LWSTrialVideoVisualizer import LWSTrialVideoVisualizer
    failed_trials = []
    for tr in subject.get_trials():
        try:
            start_trial = time.time()
            _video = LWSTrialVideoVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                             output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            plt.close('all')  # close all open figures from memory
            end_trial = time.time()
            if verbose:
                ioutils.log_and_print(
                    msg=f"\t{tr.__repr__()} Visualization:\t{(end_trial - start_trial):.2f} s",
                    log_file=subject.log_file)
        except Exception as _e:
            trace = traceback.format_exc()
            failed_trials.append((tr, trace))
            if verbose:
                ioutils.log_and_print(
                    msg=f"######\n\tFailed to visualize trial {tr.__repr__()}:\n\t{trace}\n", log_file=subject.log_file)

    end = time.time()
    if verbose:
        ioutils.log_and_print(
            msg=f"Finished visualizing all trials: {(end - start):.2f} seconds", log_file=subject.log_file)
    return failed_trials
