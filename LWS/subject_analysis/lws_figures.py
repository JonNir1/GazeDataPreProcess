import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSArrayStimulus import LWSStimulusTypeEnum
from LWS.subject_analysis.lws_instances import calculate_lws_rate as lws_rate
import Visualization.visualization_utils as visutils


def lws_rates_figure(subject: LWSSubject, proximity_thresholds: np.ndarray) -> plt.Figure:
    """
    Plot the LWS rate for each stimulus type as a function of proximity threshold.
    """
    fig = visutils.set_figure_properties(fig=None, figsize=(10, 12), tight_layout=True,
                                         title=f"LWS Rate for Varying Stimulus Types", title_height=0.94)
    bottom_ax = fig.add_subplot(212)
    bottom_ax = _draw_lws_rates(subject, proximity_thresholds, bottom_ax, proximal_fixations_only=True)

    top_ax = fig.add_subplot(211, sharex=bottom_ax)
    top_ax = _draw_lws_rates(subject, proximity_thresholds, top_ax, proximal_fixations_only=False)
    return fig


def _draw_lws_rates(subject: LWSSubject,
                    proximity_thresholds: np.ndarray, ax: plt.Axes,
                    proximal_fixations_only: bool) -> plt.Axes:
    all_trials = subject.get_all_trials()
    lws_rate_dict = {
        thrsh: {trial: lws_rate(trial, proximity_threshold=thrsh, proximal_fixations_only=proximal_fixations_only)
                for trial in all_trials} for thrsh in proximity_thresholds}
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

    ax_title = "Out of Target-Proximal Fixations" if proximal_fixations_only else "Out of All Fixations"
    x_label = "Threshold Visual Angle (°)" if proximal_fixations_only else ""
    visutils.set_axes_properties(ax=ax,
                                 ax_title=ax_title,
                                 xlabel=x_label,
                                 ylabel="LWS Rate (% fixations)",
                                 show_legend=True)
    return ax
