import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple

import constants as cnst
import experiment_config as cnfg
import Utils.io_utils as ioutils
from LWS.DataModels.LWSTrial import LWSTrial


class LWSTrialVisualizer:
    IMAGE_SUFFIX = 'png'
    VIDEO_SUFFIX = 'mp4'
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

    def __init__(self, screen_resolution: Tuple[int, int], output_directory: str = cnfg.OUTPUT_DIR):
        self.screen_resolution = screen_resolution
        self.output_directory = output_directory

    def create_gaze_figure(self, trial: LWSTrial, savefig: bool = True, **kwargs) -> plt.Figure:
        """
        Creates a figure of the raw gaze data (X, Y coordinates) during the given trial. Overlaid on the figure are
        vertical lines marking the user-inputs (triggers), and the corresponding trigger numbers are written above.
        The top part of the figure shows each sample's gaze event (fixation, saccade, blink, etc.) as a different color.

        :param trial: the trial to visualize.
        :param savefig: whether to save the figure to disk or not.

        keyword arguments:
            Gazes Related Arguments:
            - x_gaze_color: the color of the X gaze data, default is '#f03b20' (red).
            - y_gaze_color: the color of the Y gaze data, default is '#20d5f0' (light blue).

            Trigger & Event Related Arguments:
            See documentation in `__add_triggers_and_events_bar()`.

            General Arguments:
            See documentation in `__set_figure_properties()`.

        :returns: the created figure.
        """
        fig, ax = plt.subplots(tight_layout=True)

        # extract gaze data:
        dominant_eye = trial.get_subject_info().dominant_eye
        timestamps, x_gaze, y_gaze = trial.get_raw_gaze_coordinates(eye=dominant_eye)
        corrected_timestamps = timestamps - timestamps[0]  # start from 0

        # plot trial data:
        x_gaze_color = kwargs.get('x_gaze_color', '#f03b20')
        y_gaze_color = kwargs.get('y_gaze_color', '#20d5f0')
        ax.plot(corrected_timestamps, x_gaze, color=x_gaze_color, label='X (high is right)')
        ax.plot(corrected_timestamps, y_gaze, color=y_gaze_color, label='Y (high is down)')

        # add other visualizations:
        ax = self.__add_triggers_and_events_bar(ax=ax, trial=trial, **kwargs)
        fig, ax = self.__set_figure_properties(fig=fig, ax=ax,
                                               title=f"{str(trial)}",
                                               subtitle=f"Dominant Eye: {dominant_eye}",
                                               xlabel='Time (ms)', ylabel='Gaze Position (pixels)',
                                               **kwargs)

        # save figure:
        if savefig:
            subject_id = trial.get_subject_info().subject_id
            save_path = self.__get_output_full_path(subject_id, trial.trial_num, output_type='gaze_figure')
            fig.savefig(save_path, bbox_inches='tight', dpi='figure',
                        transparent=kwargs.get('transparent_figure', False))
        return fig

    def create_targets_figure(self, trial: LWSTrial, savefig: bool = True, **kwargs):
        """
        Creates a figure depicting the angular distance (visual angle) between the subject's gaze and the closest target
        during the given trial. Overlaid on the figure are vertical lines marking the user-inputs (triggers), and the
        corresponding trigger numbers are written above. Additionally, the bottom of the figure depicts gaze event for
        each sample (fixation, saccade, blink, etc.) as a different color.

        :param trial: the trial to visualize.
        :param savefig: whether to save the figure to disk or not.

        keyword arguments:
            Gazes Related Arguments:
            - line_color: the color of the angular distance line, default is '#ff0000' (red).

            Trigger & Event Related Arguments:
            See documentation in `__add_triggers_and_events_bar()`.

            General Arguments:
            See documentation in `__set_figure_properties()`.

        :returns: the created figure.
        """
        fig, ax = plt.subplots(tight_layout=True)

        # plot target distance:
        bd = trial.get_behavioral_data()
        timestamps = bd.get(cnst.MICROSECONDS).values / 1000
        corrected_timestamps = timestamps - timestamps[0]  # start from 0
        target_distance = bd.get(f"{cnst.TARGET}_{cnst.DISTANCE}").values
        ax.plot(corrected_timestamps, target_distance, color=kwargs.get('line_color', '#ff0000'), label=cnst.ANGLE)

        # add other visualizations:
        kwargs['legend_location'] = kwargs.get('legend_location', 'upper center')
        kwargs['event_bar_size'] = kwargs.get('event_bar_size', 200)
        ax = self.__add_triggers_and_events_bar(ax=ax, trial=trial, **kwargs)
        fig, ax = self.__set_figure_properties(fig=fig, ax=ax,
                                               title=f"{str(trial)}",
                                               subtitle=f"Angular Distance from Closest Target",
                                               xlabel='Time (ms)', ylabel='Visual Angle (deg)',
                                               invert_yaxis=False,
                                               **kwargs)
        # save figure:
        if savefig:
            subject_id = trial.get_subject_info().subject_id
            save_path = self.__get_output_full_path(subject_id, trial.trial_num, output_type='target_figure')
            fig.savefig(save_path, bbox_inches='tight', dpi='figure',
                        transparent=kwargs.get('transparent_figure', False))
        return fig

    def create_video(self, trial: LWSTrial, **kwargs):
        """
        Generates a video visualization of the eye-tracking data and behavioral events for the given LWSTrial.
        This video is saved to the path `self.output_directory/subject_id/trial_id.mp4`.

        :param trial: The LWSTrial object containing the raw eye-tracking data, the gaze events and behavioral data (triggers).
        :param kwargs: Additional keyword arguments for customizing the visualization parameters.

        keyword arguments:
            Stimulus Visualization:
            - show_stimulus: Whether to show the stimulus image. Defaults to True.
            - show_targets: Whether to show the targets on the stimulus image. Defaults to True.

            Trigger Visualization:
            - target_radius (int): The radius of the target circle in pixels. Defaults to 25.
            - target_edge_size (int): The width of the target circle's edge in pixels. Defaults to 4.
            - marked_target_color (Tuple[int, int, int]): The color of the marked target circle in BGR format. Defaults to (0, 0, 0) (black).
            - confirmed_target_color (Tuple[int, int, int]): The color of the confirmed target circle in BGR format. Defaults to (0, 0, 160) (dark red).

            Gaze Visualization:
            - display_gaze: Whether to display the gaze circles. Defaults to True.
            - gaze_radius (int): The radius of the gaze circle in pixels. Defaults to 10.
            - gaze_color (Tuple[int, int, int]): The color of the gaze circle in BGR format. Defaults to (255, 200, 100) (light-blue).

            Fixation Visualization:
            - display_fixations: Whether to display the fixations. Defaults to True.
            - fixation_radius (int): The radius of the fixation circle in pixels. Defaults to 45.
            - fixation_color (Tuple[int, int, int]): The color of the fixation circle in BGR format. Defaults to (40, 140, 255) (orange).
            - fixation_alpha (float): The opacity of the fixation circle. Defaults to 0.5.

        No return value
        """
        # get raw behavioral data
        timestamps, x, y = trial.get_raw_gaze_coordinates(eye='dominant')
        triggers = trial.get_triggers()
        num_samples = len(timestamps)

        # extract keyword arguments
        show_stimulus = kwargs.get('show_stimulus', True)
        show_targets = kwargs.get('show_targets', True)

        target_radius = kwargs.get('target_radius', 35)
        target_edge_size = kwargs.get('target_edge_size', 4)
        marked_target_color: Tuple[int, int, int] = kwargs.get('marked_target_color', (0, 0, 0))          # default: black
        confirmed_target_color: Tuple[int, int, int] = kwargs.get('confirmed_target_color', (0, 0, 160))  # default: dark red

        display_gaze = kwargs.get('display_gaze', True)
        gaze_radius = kwargs.get('gaze_radius', 10)
        gaze_color: Tuple[int, int, int] = kwargs.get('gaze_color', (255, 200, 100))                      # default: light-blue

        display_fixations = kwargs.get('display_fixations', True)
        fixation_radius = kwargs.get('fixation_radius', 45)
        fixation_color: Tuple[int, int, int] = kwargs.get('fixation_color', (40, 140, 255))               # default: orange
        fixation_alpha = kwargs.get('fixation_alpha', 0.5)

        # prepare video writer
        fps = round(trial.sampling_rate)
        resolution = self.screen_resolution
        save_path = self.__get_output_full_path(trial.get_subject_info().subject_id, trial.trial_num, output_type='video')
        video_writer = cv2.VideoWriter(save_path, self.FOURCC, fps, resolution)

        # prepare background image
        if show_stimulus:
            bg_img = trial.get_stimulus().get_image(color_format='BGR')
        else:
            bg_img = np.zeros((*resolution, 3), dtype=np.uint8)
        if show_targets:
            target_info = trial.get_stimulus().get_target_data()
            for _, target in target_info.iterrows():
                center_x, center_y = int(target['center_x']), int(target['center_y'])
                cv2.rectangle(bg_img, (center_x - 25, center_y - 25), (center_x + 25, center_y + 25),
                              (255, 0, 0), target_edge_size)
        bg_img = cv2.resize(bg_img, resolution)
        prev_bg_img = bg_img.copy()  # used to enable reverting to previous bg image if subject's action is undone

        # create the video:
        circle_center = np.array([np.nan, np.nan])  # to draw a circle around the target
        for i in range(num_samples):
            # get current sample data
            curr_x = int(x[i]) if not np.isnan(x[i]) else None
            curr_y = int(y[i]) if not np.isnan(y[i]) else None
            curr_trigger = int(triggers[i]) if not np.isnan(triggers[i]) else None

            # if there is a current trigger, draw it and keep it for future frames
            if curr_trigger is not None:
                if curr_trigger == cnfg.MARK_TARGET_SUCCESSFUL_TRIGGER:
                    # draw a circle around the marked target in future frames until action is confirmed/rejected
                    cv2.circle(bg_img, (curr_x, curr_y), target_radius, marked_target_color, target_edge_size)
                    circle_center = np.array([curr_x, curr_y])  # store for confirmation frame

                elif curr_trigger == cnfg.CONFIRM_TARGET_SUCCESSFUL_TRIGGER:
                    # draw a circle around the marked target in all future frames
                    cv2.circle(bg_img, circle_center, target_radius, confirmed_target_color, target_edge_size)
                    prev_bg_img = bg_img.copy()  # write over the prev_bg_img to make sure the circle stays in future frames

                elif curr_trigger == cnfg.REJECT_TARGET_SUCCESSFUL_TRIGGER:
                    # revert to previous bg image
                    bg_img = prev_bg_img.copy()

            # draw current gaze data on the frame
            gaze_img = bg_img.copy()
            if display_gaze and (curr_x is not None) and (curr_y is not None):
                cv2.circle(gaze_img, (curr_x, curr_y), gaze_radius, gaze_color, -1)

            # draw the current fixation on the frame if it exists
            fix_img = gaze_img.copy()
            if display_fixations:
                curr_t = timestamps[i]
                fixations = trial.get_gaze_events(cnst.FIXATION)
                current_fixations = list(filter(lambda f: f.start_time <= curr_t <= f.end_time, fixations))
                if len(current_fixations) > 0:
                    curr_fix = current_fixations[0]  # LWSFixationEvent
                    fix_x, fix_y = curr_fix.center_of_mass
                    cv2.circle(fix_img, (int(fix_x), int(fix_y)), fixation_radius, fixation_color, -1)

            # create a combined image of the gaze and fixation images and write it to the video
            final_img = cv2.addWeighted(fix_img, fixation_alpha, gaze_img, 1 - fixation_alpha, 0)
            video_writer.write(final_img)
        video_writer.release()

    def display_video(self, subject_id: int, trial_num: int):
        """
        Displays a video generated by `create_video` for the given subject and trial.
        Implementation adapted from https://tinyurl.com/cv2tutorial

        :raises FileNotFoundError: If the video file is not found at path `self.output_directory/subject_id/trial_num.mp4`
        """
        video_path = self.__get_output_full_path(subject_id, trial_num, output_type="video")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video file not found at {video_path}')

        window_name = f"LWS S{subject_id} T{trial_num}"
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    def __get_output_full_path(self, subject_id: int, trial_num: int, output_type: str) -> str:
        """
        Returns the full path of the output file for the given subject, trial and output type.
        Raises ValueError if the output type is not supported.
        """
        subject_dir = ioutils.create_subject_output_directory(subject_id=subject_id, output_dir=self.output_directory)
        output_type = output_type.lower()
        output_dir = ioutils.create_directory(dirname=output_type, parent_dir=subject_dir)
        if output_type == 'video':
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.VIDEO_SUFFIX}"
            return os.path.join(output_dir, filename)
        if output_type == "gaze_figure":
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.IMAGE_SUFFIX}"
            return os.path.join(output_dir, filename)
        if output_type == "target_figure":
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.IMAGE_SUFFIX}"
            return os.path.join(output_dir, filename)
        raise ValueError(f'Unsupported output type: {output_type}')

    @staticmethod
    def __add_triggers_and_events_bar(trial: LWSTrial, ax: plt.Axes, **kwargs):
        """
        Adds to the given axes two visualizations:
        1. Vertical lines and text depicting the user-inputs during the trial (i.e. triggers)
        2. A horizontal line with changing colors depicting each time-point's event type (i.e. fixation, saccade, etc.)

        :param trial: the trial to visualize
        :param ax: the axes to add the visualizations to

        keyword arguments:
            - text_size: the size of the text of the triggers (default: 12)

            Trigger Related Arguments:
            - triggers_line_color: the color of the vertical lines marking the triggers, default is 'k' (black).
            - trigger_line_width: the width of the vertical lines marking the triggers, default is 4.
            - trigger_line_style: the style of the vertical lines marking the triggers, default is ':' (dotted).

            Event Related Arguments:
            - undefined_event_color: the color of the undefined gaze events, default is '#808080' (gray).
            - blink_event_color: the color of the blink gaze events, default is '#000000' (black).
            - saccade_event_color: the color of the saccade gaze events, default is '#0000ff' (blue).
            - fixation_event_color: the color of the fixation gaze events, default is '#00ff00' (green).
            - event_bar_size: the height of the gaze event markers, default is 70 (used for `scatter`).

        Returns the axes with the added visualizations.
        """
        timestamps = trial.get_behavioral_data().get(cnst.MICROSECONDS).values / 1000
        corrected_timestamps = timestamps - timestamps[0]  # start from 0

        # Add vertical lines and text to the given axes at the given trigger times:
        triggers = trial.get_triggers()
        real_trigger_idxs = np.where((~np.isnan(triggers)) & (triggers != 0))[0]
        trigger_times = corrected_timestamps[real_trigger_idxs]
        trigger_vals = triggers[real_trigger_idxs].astype(int)
        text_size = kwargs.get('text_size', 12)
        trigger_line_color = kwargs.get('triggers_line_color', 'k')
        trigger_line_width = kwargs.get('trigger_line_width', 4)
        trigger_line_style = kwargs.get('trigger_line_style', ':')
        min_val, max_val = LWSTrialVisualizer.__get_axis_limits(ax, axis='y')  # get the min/max y values (excluding inf/nan)
        ax.vlines(x=trigger_times, ymin=0.95 * min_val, ymax=0.95 * max_val,
                  color=trigger_line_color, lw=trigger_line_width, ls=trigger_line_style)
        [ax.text(x=trigger_times[i], y=max_val, s=str(trigger_vals[i]),
                 fontsize=text_size, ha='center', va='top') for i in range(len(trigger_times))]

        # Add a horizontal bar at the top of the given axes, colored according to each timestamp's event
        event_array = trial.get_event_per_sample_array()
        undefined_event_color = kwargs.pop("undefined_event_color", "#808080")
        event_colors = np.full(shape=event_array.shape, fill_value=undefined_event_color, dtype=object)
        event_colors[event_array == cnst.BLINK] = kwargs.pop("blink_event_color", "#000000")
        event_colors[event_array == cnst.SACCADE] = kwargs.pop("saccade_event_color", "#0000ff")
        event_colors[event_array == cnst.FIXATION] = kwargs.pop("fixation_event_color", "#00ff00")
        event_bar_size = kwargs.get('event_bar_size', 70)
        event_bar_height = np.full_like(event_array, fill_value=round(max([0, min([0.95 * min_val, min_val - 1])])))
        ax.scatter(x=corrected_timestamps, y=event_bar_height, c=event_colors, s=event_bar_size, marker="s")
        return ax

    @staticmethod
    def __set_figure_properties(fig: plt.Figure, ax: plt.Axes, **kwargs):
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
            - legend_location: the location of the legend in the figure, default is 'lower center'.
            - invert_yaxis: whether to invert the Y axis or not, default is True.

        X-Axis & Y-Axis Arguments:
            - x_label: the label of the X axis, default is ''.
            - y_label: the label of the Y axis, default is ''.
            - text_size: the size of non-title text objects in the figure, default is 12.

        Returns the figure and axes with the updated properties.
        """
        # general figure properties
        figsize = kwargs.get('figsize', (16, 9))
        fig.set_size_inches(w=figsize[0], h=figsize[1])
        fig.set_dpi(kwargs.get('figure_dpi', 500))
        fig.suptitle(t=kwargs.pop('title', ''), fontsize=kwargs.get('title_size', 18), y=0.98)

        # general axis properties
        ax.set_title(label=kwargs.pop('subtitle', ''), fontsize=kwargs.pop('subtitle_size', 14))
        text_size = kwargs.get('text_size', 12)
        ax.legend(loc=kwargs.get('legend_location', 'lower center'), fontsize=text_size)

        # set x-axis properties
        _, x_axis_max = LWSTrialVisualizer.__get_axis_limits(ax, axis='x')
        ax.set_xlim(left=int(-0.01 * x_axis_max), right=int(1.01 * x_axis_max))
        jumps = 2 * np.power(10, max(0, int(np.log10(x_axis_max)) - 1))
        xticks = [int(val) for val in np.arange(int(x_axis_max)) if val % jumps == 0]
        ax.set_xticks(ticks=xticks, labels=[str(tck) for tck in xticks], rotation=45, fontsize=text_size)
        ax.set_xlabel(xlabel=kwargs.pop('xlabel', ''), fontsize=text_size)

        # set y-axis properties
        _, y_axis_max = LWSTrialVisualizer.__get_axis_limits(ax, axis='y')
        ax.set_ylim(bottom=int(-0.02 * y_axis_max), top=int(1.05 * y_axis_max))
        jumps = 2 * np.power(10, max(0, int(np.log10(y_axis_max)) - 1))
        yticks = [int(val) for val in np.arange(int(y_axis_max)) if val % jumps == 0]
        ax.set_yticks(ticks=yticks, labels=[str(tck) for tck in yticks], fontsize=text_size)
        ax.set_ylabel(ylabel=kwargs.pop('ylabel', ''), fontsize=text_size)

        if kwargs.get('invert_yaxis', True):
            # invert y-axis to match the screen coordinates:
            ax.invert_yaxis()
        return fig, ax

    @staticmethod
    def __get_axis_limits(ax: plt.Axes, axis: str) -> Tuple[float, float]:
        """ Returns the maximum and minimum values for the X or Y axis of the given plt.Axes object, excluding
        NaNs/inf """
        axis = axis.lower()
        if axis not in ['x', 'y']:
            raise ValueError(f"Invalid axis '{axis}'! Must be either 'x' or 'y'.")
        axis_idx = 0 if axis == 'x' else 1
        min_val, max_val = float('inf'), float('-inf')
        for ln in ax.lines:
            data = ln.get_data()[axis_idx]
            tmp_min = float(np.min(np.ma.masked_invalid(data)))
            tmp_max = float(np.max(np.ma.masked_invalid(data)))
            min_val = min(min_val, tmp_min)
            max_val = max(max_val, tmp_max)
        return min_val, max_val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, LWSTrialVisualizer):
            return False
        if not self.screen_resolution == other.screen_resolution:
            return False
        if not self.output_directory == other.output_directory:
            return False
        return True
