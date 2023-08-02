from enum import IntEnum


class ExperimentTriggerEnum(IntEnum):
    """ This class enumerates the triggers used in the experiment. """

    NULL = 0                # null trigger, transmitted on parallel-port after any other trigger

    START_RECORDING = 254   # eye tracker recording started
    END_RECORDING = 255     # eye tracker recording stopped

    TARGETS_ON = 13         # targets screen onset
    TARGETS_OFF = 14        # targets screen offset
    STIMULUS_ON = 15        # stimulus onset (start of trial)
    STIMULUS_OFF = 16       # stimulus offset (end of trial)

    # user input triggers:
    MARK_TARGET_SUCCESSFUL = 211
    MARK_TARGET_UNSUCCESSFUL = 212
    CONFIRM_TARGET_SUCCESSFUL = 221
    CONFIRM_TARGET_UNSUCCESSFUL = 222
    REJECT_TARGET_SUCCESSFUL = 231
    REJECT_TARGET_UNSUCCESSFUL = 232
    OTHER_KEYBOARD_INTPUT = 241
    ABORT_TRIAL = 242

