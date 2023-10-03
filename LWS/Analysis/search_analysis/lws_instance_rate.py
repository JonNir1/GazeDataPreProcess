import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LWS.DataModels.LWSTrial import LWSTrial
from LWS.DataModels.LWSSubject import LWSSubject
from GazeEvents.GazeEventEnums import GazeEventTypeEnum
from LWS.DataModels.LWSArrayStimulus import LWSStimulusTypeEnum
import Visualization.visualization_utils as visutils
import LWS.Analysis.search_analysis.identify_lws_instances as lws_instances


def lws_rates_figure(subject: LWSSubject,
                     proximity_thresholds: np.ndarray,
                     time_difference_thresholds: np.ndarray) -> plt.Figure:
    """
    Plot the LWS rate for each stimulus type as a function of proximity threshold.
    """
    nrows, ncols = 2, len(time_difference_thresholds)
    fig = visutils.set_figure_properties(fig=None, figsize=(10, 12), tight_layout=True,
                                         title=f"LWS Rate for Varying Stimulus Types\n" +
                                               "(top:\tout of all fixations\n" +
                                               "bottom:\tout of target-proximal fixations)",
                                         title_height=0.94)

    for col, td in enumerate(time_difference_thresholds):
        bottom_ax = fig.add_subplot(nrows, ncols, ncols + col + 1)
        bottom_ax = _draw_lws_rates(bottom_ax, subject, proximity_thresholds, td, proximal_fixations_only=True)

        top_ax = fig.add_subplot(nrows, ncols, col + 1, sharex=bottom_ax)
        top_ax = _draw_lws_rates(top_ax, subject, proximity_thresholds, td, proximal_fixations_only=False)
    return fig


def _draw_lws_rates(ax: plt.Axes,
                    subject: LWSSubject,
                    proximity_thresholds: np.ndarray,
                    time_difference_threshold: float,
                    proximal_fixations_only: bool) -> plt.Axes:
    lws_rate_dict = {thrsh: {trial: _calculate_lws_rate(trial,
                                                        proximity_threshold=thrsh,
                                                        proximal_fixations_only=proximal_fixations_only,
                                                        time_difference_threshold=time_difference_threshold)
                             for trial in subject.get_trials()} for thrsh in proximity_thresholds}
    lws_rate_df = pd.DataFrame.from_dict(lws_rate_dict, orient='columns')

    # extract mean of lws rate for each stimulus type:
    data_labels = ["all"] + [f"{stim_type}" for stim_type in LWSStimulusTypeEnum]
    mean_rates = np.zeros((len(data_labels), len(proximity_thresholds)))
    mean_rates[0] = lws_rate_df.mean(axis=0)
    for i, st in enumerate(LWSStimulusTypeEnum):
        stim_type_trials = list(filter(lambda tr: tr.stim_type == st, lws_rate_df.index))
        mean_rates[i + 1] = lws_rate_df.loc[stim_type_trials].mean(axis=0)

    visutils.generic_line_chart(ax=ax,
                                data_labels=data_labels,
                                xs=[proximity_thresholds for _ in range(len(data_labels))],
                                ys=[100 * mean_rates[i] for i in range(len(data_labels))])

    ax_title = f"Time-Difference Threshold:\t{time_difference_threshold:.1f} (ms)"
    x_label = "Threshold Visual Angle (°)" if proximal_fixations_only else ""
    visutils.set_axes_properties(ax=ax,
                                 ax_title=ax_title,
                                 xlabel=x_label,
                                 ylabel="LWS Rate (% fixations)",
                                 show_legend=True)
    return ax


def _calculate_lws_rate(trial: LWSTrial,
                        proximity_threshold: float,
                        time_difference_threshold: float,
                        proximal_fixations_only: bool = False) -> float:
    """
    Calculates the LWS rate for the given trial, which is the fraction of fixations that are LWS instances out of
    (a) all fixations in the trial; or (b) only the proximal fixations in the trial, depending on the value of the flag
    `proximal_fixations_only`.
    """
    # count the number of LWS instances in the trial:
    is_lws_arr = lws_instances.load_or_compute_lws_instances(trial,
                                                             proximity_threshold=proximity_threshold,
                                                             time_difference_threshold=time_difference_threshold)
    num_lws_instances = np.nansum(is_lws_arr)

    # count the number of fixations in the trial:
    fixations = trial.get_gaze_events(event_type=GazeEventTypeEnum.FIXATION)
    if proximal_fixations_only:
        fixations = list(filter(lambda f: f.visual_angle_to_closest_target <= proximity_threshold, fixations))
    num_fixations = len(fixations)

    # calculate the LWS rate:
    if num_fixations > 0:
        return num_lws_instances / num_fixations
    if num_lws_instances == 0 and num_fixations == 0:
        return np.nan
    raise ZeroDivisionError(f"num_lws_instances = {num_lws_instances},\tnum_fixations = {num_fixations}")
