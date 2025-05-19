import logging
from typing import Optional

import numpy as np

from lib.formatter import LeveledFormatter


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.val = np.nan
        self.avg = np.nan
        self.sum = np.nan
        self.count = 0

    def reset(self):
        self.val = np.nan
        self.avg = np.nan
        self.sum = np.nan
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.count == 0:
            self.sum = val * n
            self.count = n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count


def prepare_logger(
        logger_name: str, level: int = logging.INFO, log_to_console: bool = True, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Returns a logger.

    Args:
        logger_name:     str, name of the logger.
        level:           int, sets the logger level to the specified level.
        log_to_console:  bool, True to add a StreamHandler.
        log_file:        str, filename of the FileHandler.

    Returns:
        logger:          logger instance.
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = LeveledFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    formatter.set_formatter(logging.INFO, logging.Formatter('%(message)s'))
    formatter.set_formatter(logging.WARNING, logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
    formatter.set_formatter(logging.ERROR, logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))

    if log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
