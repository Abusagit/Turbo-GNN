import logging

doc = """
Lightweight logger factory with consistent formatting.
"""


def get_logger(name: str = "graph-nn-bench", level: int = logging.DEBUG) -> logging.Logger:
    """Create or fetch a configured logger.

    Args:
        name (str): Logger name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
