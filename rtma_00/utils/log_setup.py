import logging
import os
from datetime import datetime

def setup_logging(name="SignalGenerator"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, f"{name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
