import cv2
import numpy as np
from typing import Tuple

from Config import experiment_config as cnfg
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum
import Utils.io_utils as ioutils
from LWS.TrialVisualizer.LWSBaseTrialVisualizer import LWSBaseTrialVisualizer
from LWS.DataModels.LWSTrial import LWSTrial
from GazeEvents.GazeEventEnums import GazeEventTypeEnum


class LWSTrialVideoVisualizer(LWSBaseTrialVisualizer):

    __DEFAULT_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

    def __init__(self,
                 screen_resolution: Tuple[int, int] = cnfg.SCREEN_MONITOR.resolution,
                 output_directory: str = cnfg.OUTPUT_DIR,
                 codec: int = __DEFAULT_CODEC):
        super().__init__(screen_resolution, output_directory)
        self._codec = codec

    @classmethod
    def _extension(cls) -> str:
        return ioutils.VIDEO_EXTENSION

    @classmethod
    def output_dirname(cls) -> str:
        return "video"

    def visualize(self, trial: LWSTrial, should_save: bool = True, **kwargs):
        """
        Generates a video visualization of the eye-tracking data and behavioral events for the given LWSTrial.
        This video is saved to the path `self.output_directory/subject_id/trial_id.mp4`.

        :param trial: The LWSTrial object containing the raw eye-tracking data, the gaze events and behavioral data (triggers).
        :param should_save: Whether to save the result to the output directory. Defaults to True.
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
        bg_img = self._create_background_image(trial, **kwargs)
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

        # create a video writer object if should_save is True:
        video_writer = None
        if should_save:
            video_writer = cv2.VideoWriter(self.output_path(trial), self._codec,
                                           round(trial.sampling_rate), self._screen_resolution)

        # create the video:
        circle_center = np.array([np.nan, np.nan])  # to draw a circle around the target
        for i in range(num_samples):
            # get current sample data
            curr_x = int(x[i]) if not np.isnan(x[i]) else None
            curr_y = int(y[i]) if not np.isnan(y[i]) else None
            curr_trigger = int(triggers[i]) if not np.isnan(triggers[i]) else None

            # if there is a current trigger, draw it and keep it for future frames
            if curr_trigger is not None:
                if curr_trigger == ExperimentTriggerEnum.MARK_TARGET_SUCCESSFUL.value:
                    # draw a circle around the marked target in future frames until action is confirmed/rejected
                    cv2.circle(bg_img, (curr_x, curr_y), target_radius, marked_target_color, target_edge_size)
                    circle_center = np.array([curr_x, curr_y])  # store for confirmation frame

                elif curr_trigger == ExperimentTriggerEnum.CONFIRM_TARGET_SUCCESSFUL.value:
                    # draw a circle around the marked target in all future frames
                    cv2.circle(bg_img, circle_center, target_radius, confirmed_target_color, target_edge_size)
                    prev_bg_img = bg_img.copy()  # write over the prev_bg_img to make sure the circle stays in future frames

                elif curr_trigger == ExperimentTriggerEnum.REJECT_TARGET_SUCCESSFUL.value:
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
                fixations = trial.get_gaze_events(GazeEventTypeEnum.FIXATION)
                current_fixations = list(filter(lambda f: f.start_time <= curr_t <= f.end_time, fixations))
                if len(current_fixations) > 0:
                    curr_fix = current_fixations[0]  # LWSFixationEvent
                    fix_x, fix_y = curr_fix.center_of_mass
                    cv2.circle(fix_img, (int(fix_x), int(fix_y)), fixation_radius, fixation_color, -1)

            # create a combined image of the gaze and fixation images and write it to the video
            final_img = cv2.addWeighted(fix_img, fixation_alpha, gaze_img, 1 - fixation_alpha, 0)
            if video_writer is not None:
                assert should_save
                video_writer.write(final_img)

        # release the video writer object if should_save is True:
        if video_writer is not None:
            assert should_save
            video_writer.release()
