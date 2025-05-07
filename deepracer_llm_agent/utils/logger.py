import logging
import os


def setup_logger(name, level=None):
    """Sets up a logger with a default handler and formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(level or log_level)  # Allow override
    return logger
