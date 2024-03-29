import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

import Config.experiment_config as cnfg
import Utils.io_utils as ioutils
import Visualization.visualization_utils as visutils
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent


def full_pipline(name_or_id: Union[str, int],
                 save: bool = True,
                 include_subject_figures: bool = True,
                 include_trial_figures: bool = True,
                 include_trial_videos: bool = True,
                 verbose: bool = True):
    start = time.time()
    print(f"Running full pipeline for subject `{name_or_id}`...")
    subject = load_or_preprocess_subject(name_or_id=name_or_id, save=save, verbose=verbose)

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
        ioutils.print_and_log(msg=f"\nFinished processing subject {name_or_id}: {(end - start):.2f} seconds\n###############\n",
                              log_file=subject.log_file)
    return subject, subject_figures, failed_trials


def load_or_preprocess_subject(name_or_id: Union[str, int], save: bool = True, verbose: bool = True) -> LWSSubject:
    if isinstance(name_or_id, str):
        import LWS.PreProcessing as pp
        name = str(name_or_id)
        subject = pp.process_subject(subject_dir=os.path.join(cnfg.RAW_DATA_DIR, name),
                                     save_results=save,
                                     verbose=verbose,
                                     perform_subject_analysis=True,
                                     screen_monitor=cnfg.SCREEN_MONITOR,
                                     stuff_with='fixation',
                                     blink_detector_type='missing data',
                                     saccade_detector_type='engbert',
                                     drop_outlier_events=False)
    elif isinstance(name_or_id, int):
        start = time.time()
        subject_id = f"{name_or_id:03d}"
        subject = LWSSubject.from_pickle(os.path.join(cnfg.OUTPUT_DIR, f"S{subject_id}", f"LWSSubject_{subject_id}.pkl"))
        end = time.time()
        if verbose:
            ioutils.print_and_log(msg=f"Finished loading subject {str(subject)}: {(end - start):.2f} seconds",
                                  log_file=subject.log_file)
    else:
        raise ValueError(f"Invalid subject identifier: {name_or_id}")
    return subject


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

    import LWS.SubjectAnalysis.event_analysis.fixation_analysis as fixan
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

    fixation_dynamics = fixan.plot_feature_dynamics(fixation_groups, group_names, show_legend=True)
    if save:
        visutils.save_figure(fixation_dynamics,
                             full_path=os.path.join(subject_figures_dir, "fixation dynamics - all_fixations.png"))

    import LWS.SubjectAnalysis.search_analysis.lws_figures as lws_rate
    lws_rates_fig = lws_rate.plot_lws_rates(subject)
    if save:
        visutils.save_figure(lws_rates_fig,
                             full_path=os.path.join(subject_figures_dir, "lws rates.png"))

    import LWS.SubjectAnalysis.event_analysis.triggers_counts as trig
    trigger_rates = trig.plot_trigger_rates_by_block_position(subject)
    if save:
        visutils.save_figure(trigger_rates,
                             full_path=os.path.join(subject_figures_dir, "trigger rates.png"))

    import LWS.SubjectAnalysis.search_analysis.target_identification as targ_id
    angle_dist_fig = targ_id.plot_identification_angle_distribution(subject)
    if save:
        visutils.save_figure(angle_dist_fig,
                             full_path=os.path.join(subject_figures_dir, "identification angle distribution.png"))

    end = time.time()
    if verbose:
        ioutils.print_and_log(msg=f"Finished analyzing subject {subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return (saccade_distributions, all_distribution_comparison, proximal_distribution_comparison,
            distal_distribution_comparison, fixation_dynamics, lws_rates_fig, trigger_rates, angle_dist_fig)


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
                ioutils.print_and_log(
                    msg=f"\t{tr.__repr__()} Analysis:\t{(end_trial - start_trial):.2f} s", log_file=subject.log_file)
        except Exception as _e:
            trace = traceback.format_exc()
            failed_trials.append((tr, trace))
            if verbose:
                ioutils.print_and_log(
                    msg=f"######\n\tFailed to analyze trial {tr.__repr__()}:\n\t{trace}\n", log_file=subject.log_file)

    end = time.time()
    if verbose:
        ioutils.print_and_log(
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
                ioutils.print_and_log(
                    msg=f"\t{tr.__repr__()} Visualization:\t{(end_trial - start_trial):.2f} s",
                    log_file=subject.log_file)
        except Exception as _e:
            trace = traceback.format_exc()
            failed_trials.append((tr, trace))
            if verbose:
                ioutils.print_and_log(
                    msg=f"######\n\tFailed to visualize trial {tr.__repr__()}:\n\t{trace}\n", log_file=subject.log_file)

    end = time.time()
    if verbose:
        ioutils.print_and_log(
            msg=f"Finished visualizing all trials: {(end - start):.2f} seconds", log_file=subject.log_file)
    return failed_trials
