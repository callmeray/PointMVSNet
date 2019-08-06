from yacs.config import CfgNode as CN
from yacs.config import load_cfg


_C = CN()

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

_C.DATA = CN()

_C.DATA.NUM_WORKERS = 1

_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT_DIR = ""
_C.DATA.TRAIN.NUM_VIEW = 3
_C.DATA.TRAIN.NUM_VIRTUAL_PLANE = 48
_C.DATA.TRAIN.INTER_SCALE = 4.24


_C.DATA.VAL = CN()
_C.DATA.VAL = CN()
_C.DATA.VAL.ROOT_DIR = ""
_C.DATA.VAL.NUM_VIEW = 3


_C.DATA.TEST = CN()
_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT_DIR = ""
_C.DATA.TEST.NUM_VIEW = 3
_C.DATA.TEST.IMG_HEIGHT = 512
_C.DATA.TEST.IMG_WIDTH = 640
_C.DATA.TEST.NUM_VIRTUAL_PLANE = 48
_C.DATA.TEST.INTER_SCALE = 4.24

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""

_C.MODEL.EDGE_CHANNELS = ()
_C.MODEL.FLOW_CHANNELS = (64, 64, 16, 1)
_C.MODEL.NUM_VIRTUAL_PLANE = 48
_C.MODEL.IMG_BASE_CHANNELS = 8
_C.MODEL.VOL_BASE_CHANNELS = 8

_C.MODEL.VALID_THRESHOLD = 8.0

_C.MODEL.TRAIN = CN()
_C.MODEL.TRAIN.IMG_SCALES = (0.125, 0.25)
_C.MODEL.TRAIN.INTER_SCALES = (0.75, 0.375)

_C.MODEL.VAL = CN()
_C.MODEL.VAL.IMG_SCALES = (0.125, 0.25)
_C.MODEL.VAL.INTER_SCALES = (0.75, 0.375)

_C.MODEL.TEST = CN()
_C.MODEL.TEST.IMG_SCALES = (0.125, 0.25, 0.5)
_C.MODEL.TEST.INTER_SCALES = (1.0, 0.75, 0.15)

# ---------------------------------------------------------------------------- #
# Solver (optimizer)
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "RMSprop"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.RMSprop = CN()
_C.SOLVER.RMSprop.alpha = 0.9

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ""

_C.SCHEDULER.INIT_EPOCH = 2
_C.SCHEDULER.MAX_EPOCH = 2

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1


# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.LOG_PERIOD = 10
# The period to validate
_C.TRAIN.VAL_PERIOD = 0
# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
# For example, ("bn",) will freeze all batch normalization layers' weight and bias;
# And ("module:bn",) will freeze all batch normalization layers' running mean and var.
_C.TRAIN.FROZEN_PATTERNS = ()

_C.TRAIN.VAL_METRIC = "<1_cor"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10


def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template





