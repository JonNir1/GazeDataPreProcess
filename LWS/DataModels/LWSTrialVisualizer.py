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

    def create_gaze_plot(self, trial: LWSTrial, savefig: bool = True, **kwargs) -> plt.Figure:
        """
        Creates a figure of the raw gaze data (X, Y coordinates) during the given trial. Overlaid on the figure are
        vertical lines marking the user-inputs (triggers), and the corresponding trigger numbers are written above.
        The top part of the figure shows each sample's gaze event (fixation, saccade, blink, etc.) as a different color.

        :param trial: the trial to visualize.
        :param savefig: whether to save the figure to disk or not.

        keyword arguments:
            General Arguments:
            - figsize: the figure's size (width, height) in inches, default is (16, 9).
            - invert_yaxis: whether to invert the Y axis or not, default is True.
            - transparent_figure: whether to make the figure's background transparent or not, default is False.
            - figure_dpi: the figure's DPI, default is 300.

            Gazes Related Arguments:
            - x_gaze_color: the color of the X gaze data, default is '#f03b20' (red).
            - y_gaze_color: the color of the Y gaze data, default is '#20d5f0' (light blue).

            Text Related Arguments:
            - text_size: the size of non-title text objects in the figure, default is 12.
            - title_size: the size of the title text object in the figure, default is 18.
            - subtitle_size: the size of the subtitle text object in the figure, default is 14.
            - legend_location: the location of the legend in the figure, default is 'lower center'.

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

        :returns: the created figure.
        """
        # extract gaze data:
        dominant_eye = trial.get_subject_info().dominant_eye
        timestamps, x_gaze, y_gaze = trial.get_raw_gaze_coordinates(eye=dominant_eye)
        corrected_timestamps = timestamps - timestamps[0]  # start from 0
        max_time = np.nanmax(corrected_timestamps)
        max_val = np.max([np.nanmax(x_gaze), np.nanmax(y_gaze)])

        # create figure:
        figsize = kwargs.get('figsize', (16, 9))
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # plot raw gaze:
        x_gaze_color = kwargs.get('x_gaze_color', '#f03b20')
        y_gaze_color = kwargs.get('y_gaze_color', '#20d5f0')
        ax.plot(corrected_timestamps, x_gaze, color=x_gaze_color, label='X (high is right)')
        ax.plot(corrected_timestamps, y_gaze, color=y_gaze_color, label='Y (high is down)')

        # extract & plot triggers & event bar:
        triggers = trial.get_triggers()
        event_array = trial.get_event_per_sample_array()
        real_trigger_idxs = np.where((~np.isnan(triggers)) & (triggers != 0))[0]
        text_size = kwargs.get('text_size', 12)
        ax = self.__add_triggers(ax,
                                 trigger_times=corrected_timestamps[real_trigger_idxs],
                                 trigger_vals=triggers[real_trigger_idxs].astype(int),
                                 ymin=0.95 * np.min([np.nanmin(x_gaze), np.nanmin(y_gaze)]),
                                 ymax=max_val, text_size=text_size,
                                 trigger_line_color=kwargs.get('triggers_line_color', 'k'),
                                 trigger_line_width=kwargs.get('trigger_line_width', 4),
                                 trigger_line_style=kwargs.get('trigger_line_style', ':'))
        ax = self.__add_events(ax, timestamps=corrected_timestamps, events=event_array,
                               undefined_event_color=kwargs.pop("undefined_event_color", "#808080"),
                               blink_event_color=kwargs.pop("blink_event_color", "#000000"),
                               saccade_event_color=kwargs.pop("saccade_event_color", "#0000ff"),
                               fixation_event_color=kwargs.pop("fixation_event_color", "#00ff00"),
                               event_bar_size=kwargs.get('event_bar_size', 70))


        # set axes limits & ticks:
        ax = self.__set_axes_and_ticks(ax=ax, xmax=float(max_time), ymax=max_val, text_size=text_size)
        ax = self.__set_title_and_labels(ax=ax, title=f"Eye: {dominant_eye}",
                                         xlabel='Time (ms)', ylabel='Gaze Position (pixels)',
                                         title_size=kwargs.get('subtitle_size', 14), text_size=text_size)
        ax.legend(loc=kwargs.get('legend_location', 'lower center'), fontsize=text_size)
        fig.suptitle(f"{str(trial)}", fontsize=kwargs.get('title_size', 18), y=0.98)

        if kwargs.get('invert_yaxis', True):
            # invert y-axis to match the screen coordinates:
            ax.invert_yaxis()

        # save figure:
        if savefig:
            subject_id = trial.get_subject_info().subject_id
            save_path = self.__get_output_full_path(subject_id, trial.trial_num, output_type='gaze_figure')
            fig.savefig(save_path, bbox_inches='tight',
                        transparent=kwargs.get('transparent_figure', False),
                        dpi=kwargs.get('figure_dpi', 300))
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
            # TODO: add circles around the targets

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
        # TODO: add circles around the targets

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
        raise ValueError(f'Unsupported output type: {output_type}')

    @staticmethod
    def __add_triggers(ax: plt.Axes, trigger_times: np.ndarray, trigger_vals: np.ndarray,
                       ymin: float, ymax: float, **kwargs):
        """ Adds vertical lines and text to the given axes at the given trigger times. """
        text_size = kwargs.get('text_size', 12)
        trigger_line_color = kwargs.get('triggers_line_color', 'k')
        trigger_line_width = kwargs.get('trigger_line_width', 4)
        trigger_line_style = kwargs.get('trigger_line_style', ':')
        ax.vlines(x=trigger_times, ymin=ymin, ymax=ymax,
                  color=trigger_line_color, lw=trigger_line_width, ls=trigger_line_style)
        [ax.text(x=trigger_times[i], y=ymax + text_size + 1, s=str(trigger_vals[i]),
                 fontsize=text_size, ha='center', va='top') for i in range(len(trigger_times))]
        return ax

    @staticmethod
    def __add_events(ax: plt.Axes, timestamps: np.ndarray, events: np.ndarray, **kwargs):
        """ Adds a horizontal bar at the top of the given axes, colored according to each timestamp's event. """
        undefined_event_color = kwargs.pop("undefined_event_color", "#808080")
        event_colors = np.full(shape=events.shape, fill_value=undefined_event_color, dtype=object)
        event_colors[events == cnst.BLINK] = kwargs.pop("blink_event_color", "#000000")
        event_colors[events == cnst.SACCADE] = kwargs.pop("saccade_event_color", "#0000ff")
        event_colors[events == cnst.FIXATION] = kwargs.pop("fixation_event_color", "#00ff00")
        event_bar_size = kwargs.get('event_bar_size', 70)
        ax.scatter(x=timestamps, y=np.ones_like(events), c=event_colors, s=event_bar_size, marker="s")
        return ax

    @staticmethod
    def __set_axes_and_ticks(ax: plt.Axes, xmax: float, ymax: float, text_size: int = 12):
        ax.set_xlim(left=int(-0.01 * xmax), right=int(1.01 * xmax))
        xticks = [int(val) for val in np.arange(int(xmax)) if val % 1000 == 0]
        ax.set_xticks(ticks=xticks, labels=[str(tck) for tck in xticks], rotation=45, fontsize=text_size)
        ax.set_ylim(bottom=int(-0.02 * ymax), top=int(1.05 * ymax))
        yticks = [int(val) for val in np.arange(int(ymax)) if val % 200 == 0]
        ax.set_yticks(ticks=yticks, labels=[str(tck) for tck in yticks], fontsize=text_size)
        return ax

    @staticmethod
    def __set_title_and_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                               title_size: int = 14, text_size: int = 12):
        ax.set_xlabel(xlabel, fontsize=text_size)
        ax.set_ylabel(ylabel, fontsize=text_size)
        ax.set_title(title, fontsize=title_size)
        return ax

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
