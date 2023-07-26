import matplotlib.pyplot as plt

import Visualization.visualization_utils as visutils
from LWS.TrialVisualizer.LWSBaseTrialVisualizer import LWSBaseTrialVisualizer
from LWS.DataModels.LWSTrial import LWSTrial

# TODO: velocity (and acceleration?) figure with events and triggers


class LWSTrialGazeVisualizer(LWSBaseTrialVisualizer):

    @classmethod
    def output_dirname(cls) -> str:
        return "gaze_figure"

    def visualize(self, trial: LWSTrial, should_save: bool = True, **kwargs) -> plt.Figure:
        """
        Creates a figure of the raw gaze data (X, Y coordinates) during the given trial. Overlaid on the figure are
        vertical lines marking the user-inputs (triggers), and the corresponding trigger numbers are written above.
        The top part of the figure shows each sample's gaze event (fixation, saccade, blink, etc.) as a different color.

        :param trial: the trial to visualize.
        :param should_save: whether to save the figure to disk or not.

        keyword arguments:
            Gazes Related Arguments:
            - x_gaze_color: the color of the X gaze data, default is '#f03b20' (red).
            - y_gaze_color: the color of the Y gaze data, default is '#20d5f0' (light blue).

            Trigger & Event Related Arguments:
            See documentation in `self.__add_trigger_lines()` and `self.__add_events_bar()`.

            General Arguments:
            See documentation in `self.set_figure_properties()`.

        :returns: the created figure.
        """
        fig, ax = plt.subplots(tight_layout=True)

        # extract gaze data:
        timestamps, x_gaze, y_gaze, _ = trial.get_raw_gaze_data(eye='dominant')
        corrected_timestamps = timestamps - timestamps[0]  # start from 0

        # plot trial data:
        kwargs["data_labels"] = ['X (high is right)', 'Y (high is down)']
        visutils.generic_line_chart(ax=ax,
                                    xs=[corrected_timestamps, corrected_timestamps],
                                    ys=[x_gaze, y_gaze],
                                    **kwargs)

        # add other visualizations:
        ax = self._add_trigger_lines(ax=ax, trial=trial, **kwargs)
        ax = self._add_events_bar(ax=ax, trial=trial, **kwargs)
        fig, axes = self._set_figure_properties(fig=fig, ax=ax,
                                                title=f"Gaze Position over Time",
                                                subtitle=f"{str(trial)}",
                                                xlabel='Time (ms)', ylabel='Gaze Position (pixels)',
                                                invert_yaxis=True,
                                                **kwargs)
        # save figure:
        if should_save:
            visutils.save_figure(fig=fig, full_path=self.output_path(trial=trial), **kwargs)
        return fig
