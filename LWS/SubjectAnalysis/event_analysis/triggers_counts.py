import pandas as pd
import seaborn as sns

from Config.ExperimentTriggerEnum import ExperimentTriggerEnum
import Visualization.visualization_utils as visutils
from LWS.DataModels.LWSSubject import LWSSubject
from LWS.DataModels.LWSTrial import LWSTrial

DF_NAME = "trigger_counts"


def plot_trigger_rates_by_block_position(subject: LWSSubject, block_size: int = 10):
    trigger_counts = count_triggers_per_trial(subject)
    n_targets = trigger_counts["num_targets"]
    trigger_rates = trigger_counts.div(n_targets, axis=0).drop(columns=["total", "num_targets"])
    columns = trigger_rates.columns

    fig = visutils.set_figure_properties(fig=None, figsize=(18, 12), tight_layout=True,
                                         title=f"Triggers by Block Position", title_height=0.94)
    for i, col in enumerate(columns):
        ax = fig.add_subplot(2, 3, i + 1)
        sns.boxplot(ax=ax, data=trigger_rates, x=trigger_rates.index % block_size, y=col)
        ax.set_title(f"{col}")
        ax.set_xlabel("Block Position" if i >= 3 else "")
        ax.set_ylabel("Trigger per Target" if i % 3 == 0 else "")
    return fig


def count_triggers_per_trial(subject: LWSSubject) -> pd.DataFrame:
    """
    Counts the number of occurrences of each trigger type in each trial.
    Returns a DataFrame with shape (n_trials, 8) where the columns are:
        - mark: number of MARK_TARGET_SUCCESSFUL triggers
        - confirm: number of CONFIRM_TARGET_SUCCESSFUL triggers
        - reject: number of REJECT_TARGET_SUCCESSFUL triggers
        - unsuccessful: number of unsuccessful target marking triggers
        - abort: number of ABORT_TRIAL triggers
        - other: number of OTHER_KEYBOARD_INPUT triggers
        - total: total number of triggers
        - num_targets: number of targets in the trial
    """
    trigger_counts = pd.DataFrame([_trigger_count_by_category(trial) for trial in subject.get_trials()])
    trigger_counts.index.name = "trial_num"
    return trigger_counts


def _trigger_count_by_category(trial: LWSTrial) -> pd.Series:
    """ Count the number of occurrences of each trigger type in the trial. """
    trigger_counts = trial.get_trigger_counts()
    target_marking = trigger_counts[ExperimentTriggerEnum.MARK_TARGET_SUCCESSFUL]
    target_confirmation = trigger_counts[ExperimentTriggerEnum.CONFIRM_TARGET_SUCCESSFUL]
    target_rejection = trigger_counts[ExperimentTriggerEnum.REJECT_TARGET_SUCCESSFUL]
    target_unsuccessful = trigger_counts[ExperimentTriggerEnum.MARK_TARGET_UNSUCCESSFUL] + \
                          trigger_counts[ExperimentTriggerEnum.CONFIRM_TARGET_UNSUCCESSFUL] + \
                          trigger_counts[ExperimentTriggerEnum.REJECT_TARGET_UNSUCCESSFUL]
    abort_triggers = trigger_counts[ExperimentTriggerEnum.ABORT_TRIAL]
    other_triggers = trigger_counts[ExperimentTriggerEnum.OTHER_KEYBOARD_INTPUT]
    count_dict = {"mark": target_marking, "confirm": target_confirmation, "reject": target_rejection,
                  "unsuccessful": target_unsuccessful, "abort": abort_triggers, "other": other_triggers}
    count_dict["total"] = sum(count_dict.values())
    count_dict["num_targets"] = trial.num_targets
    counts = pd.Series(count_dict)
    counts.name = trial.trial_num
    return counts
