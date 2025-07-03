"""
main.py

Application startup and window management.
"""

from ui.windows.SG_main_window import SGMainWindow
from ui.windows.RTMA_main_window import RTMAMainWindow
from ui.windows.signal_generator_desk import logger


def run_main_window(app_type="sg"):
    """
    Dispatch the main window based on application type.

    Parameters
    ----------
    app_type : str
        'sg' for Signal Generator,
        'rtma' for Real-Time Modal Analyzer

    Returns
    -------
    QMainWindow
        The main application window instance.
    """
    logger.info(f"Launching app: {app_type.upper()}")
    
    if app_type == "rtma":
        return RTMAMainWindow()
    
    return SGMainWindow()
