import logging
import os
from datetime import datetime

def setup_logging(name="Application"):
    """
    Set up and return a configured logger with console and file output.

    This function initializes a logger with the specified name. The logger outputs
    messages to both the console and a timestamped log file. The log files are stored
    in a `logs` directory located in the same directory as the executing script.

    Parameters
    ----------
    name : str, optional
        The name of the logger. This name is also used in the log file name.
        Default is "Application".

    Returns
    -------
    logging.Logger
        A logger instance configured with a console stream handler and a file handler.
        Both handlers share a common formatter.

    Notes
    -----
    - Log files are named using the pattern `{name}_{YYYYMMDD_HHMMSS}.log`.
    - The `logs` directory is created if it does not exist.
    - Logging level is set to DEBUG for capturing detailed information.

    Examples
    --------
    >>> logger = setup_logging("MyApp")
    >>> logger.info("Application started.")
    >>> logger.debug("Debugging info.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(
        log_dir, f"{name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Formatter
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
