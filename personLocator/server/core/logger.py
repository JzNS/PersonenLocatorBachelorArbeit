import logging
import sys

FORMAT_STR = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger with a specific namespace.
    If the root logger is already configured, it integrates cleanly.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if the logger already has them
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT_STR, datefmt=DATE_FMT))
        logger.addHandler(handler)
        logger.propagate = True

    return logger

def get_logger(namespace: str) -> logging.Logger:
    """
    Returns a configured logger for the given namespace (e.g., 'server.network').
    """
    return setup_logger(namespace)
