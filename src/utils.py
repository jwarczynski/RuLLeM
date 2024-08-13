import logging
import time
import os
from pathlib import Path


def get_logger(log_dir, filename='webnlg', stream=True, time_in_filename=True):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = Path(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = [
        logging.FileHandler(log_dir / f"{filename}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log" if time_in_filename
                            else log_dir / f"{filename}.log", mode='w', encoding='utf-8')
    ]

    if stream:
        logger.handlers.append(logging.StreamHandler())

    for handler in logger.handlers:
        handler.setFormatter(formatter)

    logger.info(f'Logger initialized. Log directory: {log_dir}')
    return logger
