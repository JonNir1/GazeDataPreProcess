import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod

import constants as cnst
from LWS.TrialVisualizer.LWSBaseTrialVisualizer import LWSBaseTrialVisualizer
from LWS.DataModels.LWSTrial import LWSTrial
import Visualization.heatmaps as hm


class LWSTrialBaseHeatmapVisualizer(LWSBaseTrialVisualizer):

    @abstractmethod
    def _calculate_heatmap(self, trial: LWSTrial, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def visualize(self, trial: LWSTrial, should_save: bool = True, **kwargs) -> plt.Figure:
        heatmap = self._calculate_heatmap(trial=trial, **kwargs)
        heatmap[heatmap < np.mean(heatmap)] = np.nan  # remove low values

        # create RGB background image:
        bg_img = self._create_background_image(trial, **kwargs)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB for seaborn

        # overlay heatmap on background image:
        # see explanation: https://shorturl.at/gEX08
        fig, ax = plt.subplots(tight_layout=True)
        sns.heatmap(heatmap, ax=ax, cbar=False, annot=False, zorder=2,
                    cmap=kwargs.get('cmap', 'jet'),  # can also use 'hot' or 'coolwarm'
                    alpha=kwargs.get('alpha', 0.5))
        ax.imshow(bg_img, zorder=1)  # zorder=1 to put background image behind heatmap

        # configure titles and axes:
        fig, ax = self._set_figure_properties(fig=fig, ax=ax,
                                              title=self.__get_title(), subtitle=f"{str(trial)}",
                                              show_legend=False, hide_axes=True, **kwargs)
        ax.axis('off')
        # save figure:
        if should_save:
            import Visualization.visualization_utils as visutils
            visutils.save_figure(fig=fig, full_path=self.output_path(trial=trial), **kwargs)
        return fig

    def __get_title(self) -> str:
        return self.output_dirname().replace("_", " ").title()


class LWSTrialGazeHeatmapVisualizer(LWSTrialBaseHeatmapVisualizer):

    @classmethod
    def output_dirname(cls) -> str:
        return "gaze_heatmap"

    def _calculate_heatmap(self, trial: LWSTrial, **kwargs) -> np.ndarray:
        _, x_gaze, y_gaze, _ = trial.get_raw_gaze_data(eye='dominant')
        heatmap = hm.gaze_heatmap(x_gaze=x_gaze, y_gaze=y_gaze,
                                  screen_resolution=self._screen_resolution,
                                  smoothing_std=kwargs.get('smoothing_std', 10))
        return heatmap


class LWSTrialFixationsHeatmapVisualizer(LWSTrialBaseHeatmapVisualizer):

    @classmethod
    def output_dirname(cls) -> str:
        return "fixations_heatmap"

    def _calculate_heatmap(self, trial: LWSTrial, **kwargs) -> np.ndarray:
        from typing import List
        from LWS.DataModels.LWSFixationEvent import LWSFixationEvent
        fixations: List[LWSFixationEvent] = trial.get_gaze_events(event_type=LWSFixationEvent.event_type())
        heatmap = hm.fixations_heatmap(fixations=fixations, screen_resolution=self._screen_resolution)
        return heatmap

