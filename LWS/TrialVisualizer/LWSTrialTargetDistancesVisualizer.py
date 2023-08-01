
import matplotlib.pyplot as plt

import constants as cnst
import Visualization.visualization_utils as visutils
from LWS.TrialVisualizer.LWSBaseTrialVisualizer import LWSBaseTrialVisualizer
from LWS.DataModels.LWSTrial import LWSTrial


class LWSTrialTargetDistancesVisualizer(LWSBaseTrialVisualizer):

    @classmethod
    def output_dirname(cls) -> str:
        return "targets_distance"

    def visualize(self, trial: LWSTrial, should_save: bool = True, **kwargs) -> plt.Figure:
        """
        Creates a figure depicting the angular distance (visual angle) between the subject's gaze and all targets
        during the given trial. Overlaid on the figure are vertical lines marking the user-inputs (triggers), and the
        corresponding trigger numbers are written above. Additionally, the bottom of the figure depicts gaze event for
        each sample (fixation, saccade, blink, etc.) as a different color.

        :param trial: the trial to visualize.
        :param should_save: whether to save the figure to disk or not.

        keyword arguments:
            Gazes Related Arguments:
            - line_color: the color of the angular distance line, default is '#ff0000' (red).

            Trigger & Event Related Arguments:
            See documentation in `self.__add_trigger_lines()` and `self.__add_events_bar()`.

            General Arguments:
            See documentation in `self.set_figure_properties()`.

        :returns: the created figure.
        """
        fig, ax = plt.subplots(tight_layout=True)

        # extract the data
        bd = trial.get_behavioral_data()
        timestamps = bd.get(cnst.MICROSECONDS) / 1000
        corrected_timestamps = timestamps - timestamps[0]  # start from 0
        distance_columns = [col for col in bd.columns if f"{cnst.DISTANCE}_{cnst.TARGET}" in col]
        distances = bd.get(distance_columns)

        # plot the angular distance for each target:
        kwargs["data_labels"] = [f"Target {i+1}" for i in range(len(distance_columns))]
        visutils.generic_line_chart(ax=ax,
                                    xs=[corrected_timestamps for _ in range(len(distance_columns))],
                                    ys=[distances[:, i] for i in range(len(distance_columns))],
                                    **kwargs)

        # add other visualizations:
        ax = self._add_trigger_lines(ax=ax, trial=trial, **kwargs)
        ax = self._add_events_bar(ax=ax, trial=trial, **kwargs)
        fig, ax = self._set_figure_properties(fig=fig, ax=ax,
                                              title=f"Angular Distance from Targets",
                                              subtitle=f"{str(trial)}",
                                              xlabel='Time (ms)', ylabel='Visual Angle (deg)',
                                              invert_yaxis=False,
                                              show_legend=True,
                                              **kwargs)
        # save figure:
        if should_save:
            visutils.save_figure(fig=fig, full_path=self.output_path(trial=trial), **kwargs)
        return fig

