import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

import constants as cnst
from Config import experiment_config as cnfg
import Utils.io_utils as ioutils
import Visualization.visualization_utils as visutils
import Visualization.heatmaps as hm
from LWS.DataModels.LWSTrial import LWSTrial


class LWSTrialVisualizer:
    # TODO: split to separate Visualizer classes for each type of visualization
    #  (e.g. GazeVisualizer, TargetsVisualizer, etc.)
    # TODO: velocity (and acceleration?) figure with events and triggers

    IMAGE_SUFFIX = 'png'
    VIDEO_SUFFIX = 'mp4'
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

    def __init__(self, screen_resolution: Tuple[int, int] = cnfg.SCREEN_MONITOR.resolution,
                 output_directory: str = cnfg.OUTPUT_DIR):
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
            See documentation in `self.__add_trigger_lines()` and `self.__add_events_bar()`.

            General Arguments:
            See documentation in `visutils.set_figure_properties()`.

        :returns: the created figure.
        """
        fig, ax = plt.subplots(tight_layout=True)

        # extract gaze data:
        timestamps, x_gaze, y_gaze, _ = trial.get_raw_gaze_data(eye='dominant')
        corrected_timestamps = timestamps - timestamps[0]  # start from 0

        # plot trial data:
        x_gaze_color = kwargs.get('x_gaze_color', '#f03b20')
        y_gaze_color = kwargs.get('y_gaze_color', '#20d5f0')
        ax.plot(corrected_timestamps, x_gaze, color=x_gaze_color, label='X (high is right)')
        ax.plot(corrected_timestamps, y_gaze, color=y_gaze_color, label='Y (high is down)')

        # add other visualizations:
        ax = self.__add_trigger_lines(ax=ax, trial=trial, **kwargs)
        ax = self.__add_events_bar(ax=ax, trial=trial, **kwargs)
        fig, axes = visutils.set_figure_properties(fig=fig, ax=ax,
                                                   title=f"Gaze Position over Time",
                                                   subtitle=f"{str(trial)}",
                                                   xlabel='Time (ms)', ylabel='Gaze Position (pixels)',
                                                   invert_yaxis=True,
                                                   **kwargs)
        # save figure:
        if savefig:
            self.__save_figure(fig=fig, trial=trial, output_type='gaze_figure',
                               is_transparent=kwargs.get('is_transparent', False))
        return fig

    def create_targets_figure(self, trial: LWSTrial, savefig: bool = True, **kwargs) -> plt.Figure:
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
            See documentation in `self.__add_trigger_lines()` and `self.__add_events_bar()`.

            General Arguments:
            See documentation in `visutils.set_figure_properties()`.

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
        ax = self.__add_trigger_lines(ax=ax, trial=trial, **kwargs)
        ax = self.__add_events_bar(ax=ax, trial=trial, **kwargs)
        fig, ax = visutils.set_figure_properties(fig=fig, ax=ax,
                                                 title=f"Angular Distance from Closest Target",
                                                 subtitle=f"{str(trial)}",
                                                 xlabel='Time (ms)', ylabel='Visual Angle (deg)',
                                                 invert_yaxis=False,
                                                 show_legend=False,
                                                 **kwargs)
        # save figure:
        if savefig:
            self.__save_figure(fig=fig, trial=trial, output_type='targets_figure',
                               is_transparent=kwargs.get('is_transparent', False))
        return fig

    def create_heatmap(self, trial: LWSTrial, fixation_only: bool, savefig: bool = True, **kwargs) -> plt.Figure:
        # calculate heatmap:
        screen_resolution = cnfg.SCREEN_MONITOR.resolution
        if fixation_only:
            from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
            fixations = trial.get_gaze_events(cnst.FIXATION)
            fixations: List[LWSFixationEvent]
            heatmap = hm.fixations_heatmap(fixations=fixations, screen_resolution=screen_resolution)
        else:
            _, x_gaze, y_gaze, _ = trial.get_raw_gaze_data(eye='dominant')
            heatmap = hm.gaze_heatmap(x_gaze=x_gaze, y_gaze=y_gaze,
                                      screen_resolution=screen_resolution,
                                      smoothing_std=kwargs.get('smoothing_std', 10))
        heatmap[heatmap < np.mean(heatmap)] = np.nan  # remove low values

        # create RGB background image:
        bg_img = self.__create_background_image(trial, **kwargs)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB for seaborn

        # overlay heatmap on background image:
        # see explanation: https://shorturl.at/gEX08
        fig, ax = plt.subplots(tight_layout=True)
        sns.heatmap(heatmap, ax=ax, cbar=False, annot=False, zorder=2,
                    cmap=kwargs.get('cmap', 'jet'),  # can also use 'hot' or 'coolwarm'
                    alpha=kwargs.get('alpha', 0.5))
        ax.imshow(bg_img, zorder=1)  # zorder=1 to put background image behind heatmap

        # configure titles and axes:
        title = "Fixations Heatmap" if fixation_only else "Gaze Heatmap"
        fig, ax = visutils.set_figure_properties(fig=fig, ax=ax,
                                                 title=title,
                                                 subtitle=f"{str(trial)}",
                                                 show_legend=False,
                                                 show_axes=False,
                                                 **kwargs)
        ax.axis('off')

        # save figure:
        if savefig:
            hm_type = title.lower().replace(' ', '_')
            self.__save_figure(fig=fig, trial=trial, output_type=hm_type,
                               is_transparent=kwargs.get('is_transparent', False))
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
            - show_targets_color: The color of the rectangle to draw around each target. Defaults to (255, 0, 0) (blue).
            - target_edge_size (int): The thickness of the rectangle to draw around each target. Defaults to 4.

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
        timestamps, x, y, _ = trial.get_raw_gaze_data(eye='dominant')
        triggers = trial.get_triggers()
        num_samples = len(timestamps)

        # prepare background image
        bg_img = self.__create_background_image(trial, **kwargs)
        prev_bg_img = bg_img.copy()  # used to enable reverting to previous bg image if subject's action is undone

        # extract keyword arguments outside the loop to avoid unnecessary computation
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
        save_path = self.__get_full_path(trial.subject.subject_id, trial.trial_num, output_type='video')
        video_writer = cv2.VideoWriter(save_path, self.FOURCC, fps, resolution)

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
        video_path = self.__get_full_path(subject_id, trial_num, output_type="video")
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

    def __create_background_image(self, trial: LWSTrial, **kwargs) -> np.ndarray:
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
        res = self.screen_resolution
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

    def __save_figure(self, trial: LWSTrial, fig: plt.Figure, output_type: str, is_transparent: bool = False):
        subject_id = trial.subject.subject_id
        save_path = self.__get_full_path(subject_id, trial.trial_num, output_type=output_type)
        fig.savefig(save_path, bbox_inches='tight', dpi='figure', transparent=is_transparent)
        return None

    def __get_full_path(self, subject_id: int, trial_num: int, output_type: str) -> str:
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
        if output_type == "targets_figure":
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.IMAGE_SUFFIX}"
            return os.path.join(output_dir, filename)
        if output_type == "gaze_heatmap":
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.IMAGE_SUFFIX}"
            return os.path.join(output_dir, filename)
        if output_type == "fixations_heatmap":
            filename = f"T{trial_num:03d}.{LWSTrialVisualizer.IMAGE_SUFFIX}"
            return os.path.join(output_dir, filename)
        raise ValueError(f'Unsupported output type: {output_type}')

    @staticmethod
    def __add_trigger_lines(trial: LWSTrial, ax: plt.Axes, **kwargs) -> plt.Axes:
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
        min_val, max_val = visutils.get_line_axis_limits(ax, axis='y')  # get the min/max y values (excluding inf/nan)

        triggers = trial.get_triggers()
        real_trigger_idxs = np.where((~np.isnan(triggers)) & (triggers != 0))[0]
        trigger_times = corrected_timestamps[real_trigger_idxs]
        trigger_vals = triggers[real_trigger_idxs].astype(int)

        # Add vertical lines and text to the given axes at the given trigger times:
        text_size = kwargs.get('text_size', 12)
        trigger_line_color = kwargs.get('triggers_line_color', 'k')
        trigger_line_width = kwargs.get('trigger_line_width', 4)
        trigger_line_style = kwargs.get('trigger_line_style', ':')
        ax.vlines(x=trigger_times, ymin=0.95 * min_val, ymax=0.95 * max_val,
                  color=trigger_line_color, lw=trigger_line_width, ls=trigger_line_style)
        [ax.text(x=trigger_times[i], y=max_val, s=str(trigger_vals[i]),
                 fontsize=text_size, ha='center', va='top') for i in range(len(trigger_times))]
        return ax

    @staticmethod
    def __add_events_bar(trial: LWSTrial, ax: plt.Axes, **kwargs) -> plt.Axes:
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
        event_array = trial.get_event_per_sample_array()
        undefined_event_color = kwargs.pop("undefined_event_color", "#808080")
        event_colors = np.full(shape=event_array.shape, fill_value=undefined_event_color, dtype=object)
        event_colors[event_array == cnst.BLINK] = kwargs.pop("blink_event_color", "#000000")
        event_colors[event_array == cnst.SACCADE] = kwargs.pop("saccade_event_color", "#0000ff")
        event_colors[event_array == cnst.FIXATION] = kwargs.pop("fixation_event_color", "#00ff00")

        # Add a horizontal scatter plot to the axes, depicting the events:
        min_val, max_val = visutils.get_line_axis_limits(ax, axis='y')  # get the min/max y values (excluding inf/nan)
        event_bar_width = kwargs.get('event_bar_width', 50)
        event_bar_height = np.full_like(event_array, fill_value=round(max([0, min([0.95 * min_val, min_val - 1])])))
        ax.scatter(x=corrected_timestamps, y=event_bar_height, c=event_colors, s=event_bar_width, marker="s")
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
