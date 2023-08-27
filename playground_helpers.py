import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import Config.experiment_config as cnfg
import Utils.io_utils as ioutils
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from GazeEvents.SaccadeEvent import SaccadeEvent
from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
import Visualization.visualization_utils as visutils


def full_pipline(name: str, save: bool = True,
                 run_analysis: bool = True,
                 run_visualization: bool = True,
                 verbose: bool = True):
    start = time.time()
    print(f"Processing subject `{name}`...")
    subject = process_subject(name=name, save=save, verbose=verbose)
    subject = load_subject(subject_id=subject.subject_id, verbose=verbose)
    subject_analysis = None
    failed_analysis_trials = []
    failed_visualization_trials = []
    if run_analysis:
        subject_analysis = analyze_subject(subject=subject, save=save, verbose=verbose)
        failed_analysis_trials = analyze_all_trials(subject=subject, save=save, verbose=verbose)
    if run_visualization:
        failed_visualization_trials = visualize_all_trials(subject=subject, save=save, verbose=verbose)
    failed_trials = failed_analysis_trials + failed_visualization_trials
    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"\nFinished processing subject {name}: {(end - start):.2f} seconds\n###############\n",
                              log_file=subject.log_file)
    return subject, subject_analysis, failed_trials


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


def analyze_subject(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    import Utils.io_utils as ioutils
    import LWS.subject_analysis.trial_summary as trsum
    import Visualization.saccade_analysis as sacan
    import LWS.subject_analysis.fixation_analysis as fixan
    import LWS.subject_analysis.lws_figures as lws_fig

    trials = subject.get_all_trials()
    all_saccades: List[SaccadeEvent] = [s for tr in trials for s in tr.get_gaze_events(GazeEventTypeEnum.SACCADE)]
    all_fixations: List[LWSFixationEvent] = [f for tr in trials for f in tr.get_gaze_events(GazeEventTypeEnum.FIXATION)]
    subject_figures_dir = ioutils.create_directory(dirname="subject_figures", parent_dir=subject.output_dir)

    trial_summary = trsum.summarize_all_trials(trials)
    if save:
        trial_summary.to_pickle(os.path.join(subject.output_dir, "trials_summary.pkl"))

    saccade_distributions = sacan.distributions_figure(all_saccades, ignore_outliers=True,
                                                       title="Saccades Property Distributions", show_legend=True)
    if save:
        visutils.save_figure(saccade_distributions,
                             full_path=os.path.join(subject_figures_dir, "saccade distributions.png"))

    fixation_distributions = fixan.distributions_figure(all_fixations, ignore_outliers=True,
                                                        title="Fixations Property Distributions", show_legend=True)
    if save:
        visutils.save_figure(fixation_distributions,
                             full_path=os.path.join(subject_figures_dir, "fixation distributions.png"))

    fixation_dynamics = fixan.dynamics_figure(all_fixations, ignore_outliers=True,
                                              title="Fixations Temporal Dynamics", show_legend=True)
    if save:
        visutils.save_figure(fixation_dynamics,
                             full_path=os.path.join(subject_figures_dir, "fixation dynamics.png"))

    fixation_proximity_comparison = fixan.target_proximal_comparison(all_fixations, ignore_outliers=True,
                                                                     title="Fixations Proximity Comparison",
                                                                     show_legend=True)
    if save:
        visutils.save_figure(fixation_proximity_comparison,
                             full_path=os.path.join(subject_figures_dir, "target-proximal fixation comparison.png"))

    lws_rates = lws_fig.lws_rates_figure(subject, proximity_thresholds=np.arange(0.1 * cnfg.THRESHOLD_VISUAL_ANGLE,
                                                                                 1.2 * cnfg.THRESHOLD_VISUAL_ANGLE,
                                                                                 0.1 * cnfg.THRESHOLD_VISUAL_ANGLE))
    if save:
        visutils.save_figure(lws_rates,
                             full_path=os.path.join(subject_figures_dir, "lws rates.png"))

    end = time.time()
    if verbose:
        ioutils.log_and_print(msg=f"Finished analyzing subject {subject.subject_id}: {(end - start):.2f} seconds",
                              log_file=subject.log_file)
    return trial_summary, saccade_distributions, fixation_distributions, fixation_dynamics, fixation_proximity_comparison


def analyze_all_trials(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    from LWS.TrialVisualizer.LWSTrialGazeVisualizer import LWSTrialGazeVisualizer
    from LWS.TrialVisualizer.LWSTrialTargetDistancesVisualizer import LWSTrialTargetDistancesVisualizer
    from LWS.TrialVisualizer.LWSTrialHeatmapVisualizer import LWSTrialGazeHeatmapVisualizer, LWSTrialFixationsHeatmapVisualizer

    failed_trials = []
    for tr in subject.get_all_trials():
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


def visualize_all_trials(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    from LWS.TrialVisualizer.LWSTrialVideoVisualizer import LWSTrialVideoVisualizer
    failed_trials = []
    for tr in subject.get_all_trials():
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
