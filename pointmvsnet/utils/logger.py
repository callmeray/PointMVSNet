# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import sys
import time
from path import Path


def setup_logger(name, save_dir, prefix="", timestamp=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else ""
        prefix = "." + prefix if prefix else ""
        log_file = Path(save_dir) / "log{}.txt".format(prefix + timestamp)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def shutdown_logger(logger):
    logger.handlers = []
