import logging
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

def get_logger(name="sfm"):

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = log_dir / f".{timestamp}.log"

    if not logger.handlers:
        handler = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_timing(func):

    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()

        logger.info(
            f"{func.__name__} took {end - start:.4f} seconds"
        )

        return result

    return wrapper

def timed_call(name, func, *args, **kwargs):

    start = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = time.perf_counter() - start

    logger.info(
        f"{name} took {elapsed:.4f} seconds"
    )

    return result

@contextmanager
def log_time(name):

    start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - start

    logger.info(
        f"{name} took {elapsed:.4f} seconds"
    )

logger = get_logger()