import os
import time
import matplotlib.pyplot as plt

import constants as cnst
import Config.experiment_config as cnfg
from LWS.DataModels.LWSSubject import LWSSubject
import Visualization.visualization_utils as visutils


def full_pipline(name: str, save: bool = True, verbose: bool = True):
    start = time.time()
    print(f"Processing subject `{name}`...")
    subject = process_subject(name=name, save=save, verbose=verbose)
    subject = load_subject(subject_id=subject.subject_id, verbose=verbose)
    subject_analysis = analyze_subject(subject=subject, save=save, verbose=verbose)
    failed_trials = visualize_all_trials(subject=subject, save=save, verbose=verbose)
    end = time.time()
    if verbose:
        print(f"\nFinished processing subject {name}: {(end - start):.2f} seconds\n###############\n")
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
        print(f"Finished preprocessing subject `{name}`: {(end - start):.2f} seconds")
    return subject


def load_subject(subject_id: int, verbose: bool = True) -> LWSSubject:
    start = time.time()
    subject_id = f"{subject_id:03d}"
    subdir = f"S{subject_id}"
    subject = LWSSubject.from_pickle(os.path.join(cnfg.OUTPUT_DIR, subdir, f"LWSSubject_{subject_id}.pkl"))
    end = time.time()
    if verbose:
        print(f"Finished loading subject {subject_id:03d}: {(end - start):.2f} seconds")
    return subject


def analyze_subject(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    import Utils.io_utils as ioutils
    import LWS.analysis_scripts.trial_summary as trsum
    import Visualization.saccade_analysis as sacan
    import LWS.analysis_scripts.fixation_analysis as fixan

    trials = subject.get_all_trials()
    all_saccades = [s for tr in trials for s in tr.get_gaze_events(cnst.SACCADE)]
    all_fixations = [f for tr in trials for f in tr.get_gaze_events(cnst.FIXATION)]

    trial_summary = trsum.summarize_all_trials(trials)
    saccade_distributions = sacan.distributions_figure(all_saccades, ignore_outliers=True,
                                                       title="Saccades Property Distributions", show_legend=True)
    fixation_distributions = fixan.distributions_figure(all_fixations, ignore_outliers=True,
                                                        title="Fixations Property Distributions", show_legend=True)
    fixation_dynamics = fixan.dynamics_figure(all_fixations, ignore_outliers=True,
                                              title="Fixations Temporal Dynamics", show_legend=True)
    fixation_proximity_comparison = fixan.target_proximal_comparison(all_fixations, ignore_outliers=True,
                                                                     title="Fixations Proximity Comparison",
                                                                     show_legend=True)
    if save:
        trial_summary.to_pickle(os.path.join(subject.output_dir, "trials_summary.pkl"))
        subject_figures_dir = ioutils.create_directory(dirname="subject_figures", parent_dir=subject.output_dir)
        visutils.save_figure(saccade_distributions,
                             full_path=os.path.join(subject_figures_dir, "saccade distributions.png"))
        visutils.save_figure(fixation_distributions,
                             full_path=os.path.join(subject_figures_dir, "fixation distributions.png"))
        visutils.save_figure(fixation_dynamics, full_path=os.path.join(subject_figures_dir, "fixation dynamics.png"))
        visutils.save_figure(fixation_proximity_comparison,
                             full_path=os.path.join(subject_figures_dir, "target-proximal fixation comparison.png"))
    end = time.time()
    if verbose:
        print(f"Finished analyzing subject {subject.subject_id}: {(end - start):.2f} seconds")
    return trial_summary, saccade_distributions, fixation_distributions, fixation_dynamics, fixation_proximity_comparison


def visualize_all_trials(subject: LWSSubject, save: bool = False, verbose: bool = True):
    start = time.time()
    from LWS.TrialVisualizer.LWSTrialGazeVisualizer import LWSTrialGazeVisualizer
    from LWS.TrialVisualizer.LWSTrialTargetDistancesVisualizer import LWSTrialTargetDistancesVisualizer
    from LWS.TrialVisualizer.LWSTrialHeatmapVisualizer import LWSTrialGazeHeatmapVisualizer, LWSTrialFixationsHeatmapVisualizer
    from LWS.TrialVisualizer.LWSTrialVideoVisualizer import LWSTrialVideoVisualizer

    failed_trials = []
    for tr in subject.get_all_trials():
        try:
            start_trial = time.time()
            gaze = LWSTrialGazeVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                          output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            targets = LWSTrialTargetDistancesVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                        output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            gaze_heatmap = LWSTrialGazeHeatmapVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                         output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            fixation_heatmap = LWSTrialFixationsHeatmapVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                                                  output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            video = LWSTrialVideoVisualizer(screen_resolution=cnfg.SCREEN_MONITOR.resolution,
                                            output_directory=cnfg.OUTPUT_DIR).visualize(tr, should_save=save)
            plt.close('all')  # close all open figures from memory
            end_trial = time.time()
            if verbose:
                print(f"\t{tr.__repr__()}:\t{(end_trial - start_trial):.2f} s")
        except Exception as e:
            failed_trials.append((tr, e))
            if verbose:
                print(f"Failed to visualize trial {tr.__repr__()}:\n{e}\n")

    end = time.time()
    if verbose:
        print(f"Finished visualizing all trials: {(end - start):.2f} seconds")
    return failed_trials
