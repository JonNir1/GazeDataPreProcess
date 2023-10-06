import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Config.experiment_config as cnfg
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSArrayStimulus import LWSStimulusTypeEnum
import Visualization.visualization_utils as visutils
import LWS.SubjectAnalysis.search_analysis.identify_lws_instances as identify_lws

_FRAC_ALLOWED_NANS = 0.5


def plot_lws_rates(subject: LWSSubject) -> plt.Figure:
    """
    Plot the LWS rate for each stimulus type as a function of proximity threshold.
    Top row of the figure is the LWS rate out of all fixations, bottom row is the LWS rate out of target-proximal fixations.
    """
    # load the LWS rate dataframes:
    lws_rates_all__fixations = _load_lws_rate_df(subject, proximal_fixations_only=False)
    lws_rates__proximal_fixations = _load_lws_rate_df(subject, proximal_fixations_only=True)

    td_thresholds__all = lws_rates_all__fixations.columns.get_level_values("time_difference_threshold").unique()
    td_thresholds__proximal = lws_rates__proximal_fixations.columns.get_level_values("time_difference_threshold").unique()
    assert np.all(td_thresholds__all == td_thresholds__proximal)  # make sure the time difference thresholds are the same

    fig, axes = plt.subplots(nrows=2, ncols=len(td_thresholds__all),
                             figsize=(27, 15), tight_layout=True,
                             sharex='col', sharey='row')
    for col, td in enumerate(td_thresholds__all):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]

        try:
            top_ax = _draw_lws_rates(top_ax, lws_rates_all__fixations, td)
        except KeyError:
            pass
        try:
            bottom_ax = _draw_lws_rates(axes[1, col], lws_rates__proximal_fixations, td)
        except KeyError:
            pass

        visutils.set_axes_properties(ax=top_ax,
                                     ax_title=f"Δt Threshold: {td:.2f} ms\n" +
                                              f"(saccade duration {cnfg.TIME_DIFF_PERCENTILE_THRESHOLDS[col]} percentile)",
                                     show_legend=(col == len(td_thresholds__all) - 1))
        visutils.set_axes_properties(ax=bottom_ax,
                                     xlabel="Threshold Visual Angle (°)")

    fig = visutils.set_figure_properties(fig=fig, figsize=(27, 12), tight_layout=True,
                                         title=f"LWS Rate for Varying Stimulus Types\n" +
                                               "(top: out of all fixations ; bottom: out of target-proximal fixations)",
                                         title_height=0.98)
    return fig


def _draw_lws_rates(ax: plt.Axes,
                    lws_rates: pd.DataFrame,
                    td_threshold: float,
                    frac_nans: float = _FRAC_ALLOWED_NANS) -> plt.Axes:
    # filter by time difference threshold:
    lws_rate_for_td = lws_rates.loc[:, lws_rates.columns.get_level_values(1) == td_threshold]
    lws_rate_for_td.columns = lws_rate_for_td.columns.droplevel(1)  # drop the `Δt threshold` level
    if lws_rate_for_td.empty:
        raise KeyError(f"No data for Δt threshold {td_threshold}")

    # extract mean of lws rate for each stimulus type:
    data_labels = ["all"] + [f"{stim_type}" for stim_type in LWSStimulusTypeEnum]
    mean_rates = pd.DataFrame(np.nan, index=data_labels, columns=lws_rate_for_td.columns.values)
    sem_rates = pd.DataFrame(np.nan, index=data_labels, columns=lws_rate_for_td.columns.values)

    # calculate mean of lws rate for all stimuli types:
    means, sems = _calc_mean_rate_and_sem(rates_df=lws_rate_for_td, frac_nans=frac_nans)
    mean_rates.loc[data_labels[0], means.index] = means
    sem_rates.loc[data_labels[0], sems.index] = sems

    # calculate mean of lws rate for each stimulus type:
    for i, st in enumerate(LWSStimulusTypeEnum):
        stim_type_trials = list(filter(lambda tr: tr.stim_type == st, lws_rate_for_td.index))
        means, sems = _calc_mean_rate_and_sem(rates_df=lws_rate_for_td.loc[stim_type_trials],
                                              frac_nans=frac_nans)
        mean_rates.loc[data_labels[i + 1], means.index] = means
        sem_rates.loc[data_labels[i + 1], sems.index] = sems

    visutils.generic_line_chart(ax=ax,
                                data_labels=data_labels,
                                xs=[lws_rate_for_td.columns.values for _ in range(len(data_labels))],
                                ys=[100 * mean_rates.iloc[i] for i in range(len(data_labels))],
                                sems=[100 * sem_rates.iloc[i] for i in range(len(data_labels))])
    return ax


def _load_lws_rate_df(subject: LWSSubject, proximal_fixations_only: bool) -> pd.DataFrame:
    """
    Load the LWS rate dataframe for a specific subject, for either all fixations or only proximal fixations.

    :param subject: the subject to load the LWS rate dataframe for
    :param proximal_fixations_only: whether to load the LWS rate dataframe for proximal fixations only

    :returns: the LWS rate dataframe
    """
    df_name = identify_lws.RATES_DF_BASE_NAME + ("_proximal_fixations" if proximal_fixations_only else "_all_fixations")
    lws_rate_df = subject.get_dataframe(df_name)
    if lws_rate_df is None:
        raise AttributeError(f"Subject {subject} does not have a dataframe named {df_name}")
    return lws_rate_df


def _calc_mean_rate_and_sem(rates_df: pd.DataFrame, frac_nans: float = _FRAC_ALLOWED_NANS) -> (pd.Series, pd.Series):
    """
    Calculate the mean & SEM of LWS rates for each proximity threshold, ignoring proximity thresholds with too many NaN
    trials.

    :param rates_df: a (num trials × num proximity thresholds) dataframe containing the LWS rate for each
        (trial, proximity threshold) pair
    :param frac_nans: the fraction of NaN trials allowed for each proximity threshold, must be between 0 and 1

    :returns: two pandas Series containing the mean and SEM of LWS rates for each proximity threshold

    :raises ValueError: if `frac_nans` is not between 0 and 1
    """
    if frac_nans < 0 or frac_nans > 1:
        raise ValueError(f"Invalid `frac_nans` value: {frac_nans}")
    num_trials = len(rates_df)
    num_nans_for_prox_thresh = rates_df.isna().sum(axis=0)  # number of NaN trials for each proximity threshold
    is_valid_proximity_threshold = num_nans_for_prox_thresh < num_trials * frac_nans
    mean_of_valids = rates_df.loc[:, is_valid_proximity_threshold].mean(axis=0)
    sem_of_valids = rates_df.loc[:, is_valid_proximity_threshold].sem(axis=0)
    return mean_of_valids, sem_of_valids
