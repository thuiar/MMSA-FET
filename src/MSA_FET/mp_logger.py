import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path


def init_log_worker(queue, name='FET-Subprocess'):
    qh = QueueHandler(queue)
    logger = logging.getLogger(name)
    logger.addHandler(qh)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.debug("Logger initialized.")
    return logger
    
def init_log_listener(
    queue : multiprocessing.Queue,
    log_dir : str | Path = Path.home() / '.MMSA-FET/log',
    stream_level : int = logging.INFO
):
    fh = RotatingFileHandler(log_dir / 'MSA-FET.log', maxBytes=2e7, backupCount=2)
    fh_formatter = logging.Formatter(
        fmt = '%(asctime)s - %(name)s - PID:%(process)d [%(levelname)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(stream_level)
    ch_formatter = logging.Formatter(
        fmt = '%(asctime)s - %(name)s - PID:%(process)d [%(levelname)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(ch_formatter)
    listener = QueueListener(queue, ch, fh)
    return listener
