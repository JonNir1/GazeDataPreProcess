import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

import constants as cnst
from Config import experiment_config as cnfg
import Utils.io_utils as ioutils
import Visualization.visualization_utils as visutils
from LWS.DataModels.LWSTrial import LWSTrial


class LWSBaseTrialVisualizer(ABC):

    def __init__(self, screen_resolution: Tuple[int, int] = cnfg.SCREEN_MONITOR.resolution,
                 output_directory: str = cnfg.OUTPUT_DIR):
        self._screen_resolution = screen_resolution
        self._output_directory = output_directory

    @classmethod
    def _extension(cls) -> str:
        return ioutils.IMAGE_EXTENSION

    @classmethod
    @abstractmethod
    def output_dirname(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def visualize(self, trial: LWSTrial, **kwargs):
        """
        Visualizes the given trial and saves the output to a file.
        """
        raise NotImplementedError

    def output_path(self, trial: LWSTrial) -> str:
        """
        Returns the full path of the output file for the given subject, trial and output type.
        """
        subject_id = trial.subject.subject_id
        trial_num = trial.trial_num
        subject_dir = ioutils.create_subject_output_directory(subject_id=subject_id, output_dir=self._output_directory)
        output_dir = ioutils.create_directory(dirname=self.output_dirname(), parent_dir=subject_dir)
        filename = ioutils.get_filename(name=f"T{trial_num:03d}", extension=self._extension())
        return os.path.join(output_dir, filename)

    def _get_full_path(self, subject_id: int, trial_num: int) -> str:
        """
        Returns the full path of the output file for the given subject, trial and output type.
        """
        subject_dir = ioutils.create_subject_output_directory(subject_id=subject_id, output_dir=self._output_directory)
        output_dir = ioutils.create_directory(dirname=self.output_dirname(), parent_dir=subject_dir)
        filename = ioutils.get_filename(name=f"T{trial_num:03d}", extension=self._extension())
        return os.path.join(output_dir, filename)

    def _create_background_image(self, trial: LWSTrial, **kwargs) -> np.ndarray:
        """
        Creates the background image for the given trial:
        - If `show_stimulus` is True, the stimulus image is used as the background image. Otherwise, a black image is used.
        - If `show_targets` is True, a rectangle is drawn around each target in the background image.

        Keyword Arguments:
        - show_stimulus: Whether to show the stimulus image as the background image. Default is True.
        - show_targets: Whether to draw a rectangle around each target in the background image. Default is True.
        - show_targets_color: The color of the rectangle to draw around each target. Default is (255, 0, 0) (blue).
        - target_edge_size: The thickness of the rectangle to draw around each target. Default is 4.

        :return: The background image as a numpy array of shape (height, width, 3)
        """
        res = self._screen_resolution
        if kwargs.pop("show_stimulus", True):
            bg_img = trial.get_stimulus().get_image(color_format='BGR')
        else:
            bg_img = np.zeros((*res, 3), dtype=np.uint8)
        if kwargs.pop("show_targets", True):
            color = kwargs.pop("show_targets_color", (255, 0, 0))  # BGR format
            edge_size = int(kwargs.pop("target_edge_size", 4))
            target_info = trial.get_stimulus().get_target_data()
            for _, target in target_info.iterrows():
                center_x, center_y = int(target['center_x']), int(target['center_y'])
                cv2.rectangle(bg_img, (center_x - 25, center_y - 25), (center_x + 25, center_y + 25), color, edge_size)
        bg_img = cv2.resize(bg_img, res)
        return bg_img

    @staticmethod
    def _add_trigger_lines(trial: LWSTrial, ax: plt.Axes, **kwargs) -> plt.Axes:
        """
        Adds to the given Axes a set of vertical lines and text depicting the user-inputs during the trial (i.e. triggers)
        :param trial: the trial to visualize
        :param ax: the axes to add the visualizations to

        keyword arguments:
            - text_size: the size of the text of the triggers (default: 12)
            - triggers_line_color: the color of the vertical lines marking the triggers, default is 'k' (black).
            - trigger_line_width: the width of the vertical lines marking the triggers, default is 4.
            - trigger_line_style: the style of the vertical lines marking the triggers, default is ':' (dotted).

        Returns the axes with the added visualizations.
        """
        # Extract the relevant data from the trial:
        timestamps = trial.get_behavioral_data().get(cnst.MICROSECONDS).values / 1000
        corrected_timestamps = timestamps - timestamps[0]  # start from 0
        triggers = trial.get_triggers()
        real_trigger_idxs = np.where((~np.isnan(triggers)) & (triggers != 0))[0]
        trigger_times = corrected_timestamps[real_trigger_idxs]
        trigger_vals = triggers[real_trigger_idxs].astype(int)

        # Add vertical lines and text to the given axes at the given trigger times:
        text_size = kwargs.get('text_size', 12)
        trigger_line_color = kwargs.get('triggers_line_color', 'k')
        trigger_line_width = kwargs.get('trigger_line_width', 4)
        trigger_line_style = kwargs.get('trigger_line_style', ':')
        min_val, max_val = ax.get_ylim()
        ax.vlines(x=trigger_times, ymin=0.95 * min_val, ymax=0.92 * max_val,
                  color=trigger_line_color, lw=trigger_line_width, ls=trigger_line_style)
        [ax.text(x=trigger_times[i], y=0.93 * max_val, s=str(trigger_vals[i]),
                 fontsize=text_size, ha='center', va='bottom') for i in range(len(trigger_times))]
        return ax

    @staticmethod
    def _add_events_bar(trial: LWSTrial, ax: plt.Axes, **kwargs) -> plt.Axes:
        # TODO: implement this using a second axes, and not a second plot on the same axes.
        # the second axes should be aligned with the first one, but with no ticks and no labels.
        """
        Adds to the given axes a horizontal line with changing colors depicting each time-point's event-type
        (i.e. fixation, saccade, etc.). The colors are defined by color mappings provided as keyword arguments.
        :param trial: the trial to visualize
        :param ax: the axes to add the visualizations to

        keyword arguments:
            - text_size: the size of the text of the triggers (default: 12)
            - undefined_event_color: the color of the undefined gaze events, default is '#808080' (gray).
            - blink_event_color: the color of the blink gaze events, default is '#000000' (black).
            - saccade_event_color: the color of the saccade gaze events, default is '#0000ff' (blue).
            - fixation_event_color: the color of the fixation gaze events, default is '#00ff00' (green).
            - event_bar_width: the height of the gaze event markers, default is 70 (used for `scatter`).

        Returns the axes with the added visualizations.
        """

        # Extract the relevant data from the trial:
        timestamps = trial.get_behavioral_data().get(cnst.MICROSECONDS).values / 1000
        corrected_timestamps = timestamps - timestamps[0]  # start from 0

        # create an array of colors per sample, depicting the events:
        event_array = trial.get_event_per_sample()
        undefined_event_color = kwargs.pop("undefined_event_color", "#808080")
        event_colors = np.full(shape=len(event_array), fill_value=undefined_event_color, dtype=object)
        event_colors[event_array == cnst.BLINK] = kwargs.pop("blink_event_color", "#000000")
        event_colors[event_array == cnst.SACCADE] = kwargs.pop("saccade_event_color", "#0000ff")
        event_colors[event_array == cnst.FIXATION] = kwargs.pop("fixation_event_color", "#00ff00")

        # Add a horizontal scatter plot to the axes, depicting the events:
        min_val, _ = ax.get_ylim()
        event_bar_width = kwargs.get('event_bar_width', 50)
        event_bar_height = np.full_like(event_array, fill_value=round(min_val - 1))
        ax.scatter(x=corrected_timestamps, y=event_bar_height, c=event_colors, s=event_bar_width, marker="s")
        return ax

    @staticmethod
    def _set_figure_properties(fig: plt.Figure, ax: plt.Axes, **kwargs):
        """
        Sets the properties of the given figure and axes according to the given keyword arguments:

        Figure Arguments:
            - figsize: the figure's size (width, height) in inches, default is (16, 9).
            - figure_dpi: the figure's DPI, default is 300.
            - title: the figure's title, default is ''.
            - title_size: the size of the title text object in the figure, default is 18.

        Axes Arguments:
            - subtitle: the figure's subtitle, default is ''.
            - subtitle_size: the size of the subtitle text object in the figure, default is 14.
            - show_legend: whether to show the legend or not, default is True.
            - legend_location: the location of the legend in the figure, default is 'lower center'.
            - invert_yaxis: whether to invert the Y axis or not, default is False.

        X-Axis & Y-Axis Arguments:
            - hide_axes: whether to hide the x,y axes or not, default is False.
            - xlabel: the label of the X axis, default is ''.
            - ylabel: the label of the Y axis, default is ''.
            - text_size: the size of non-title text objects in the figure, default is 12.

        Returns the figure and axes with the updated properties.
        """
        fig = visutils.set_figure_properties(fig=fig, **kwargs)
        ax = visutils.set_axes_properties(ax=ax, ax_title=kwargs.pop("subtitle", ""),
                                          xlabel=kwargs.pop("xlabel", ""), ylabel=kwargs.pop("ylabel", ""),
                                          **kwargs)
        return fig, ax

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if not self._screen_resolution == other._screen_resolution:
            return False
        if not self._output_directory == other._output_directory:
            return False
        return True
