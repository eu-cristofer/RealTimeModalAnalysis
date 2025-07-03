"""
window_selector

Application startup and window management.
"""
from rtma.utils.logger import get_logger
from rtma.ui.windows.SG_main_window import SGMainWindow
# from rtma.ui.windows.RTMA_main_window import RTMAMainWindow

log = get_logger(__name__)

def run_main_window(app_type="sg"):
    """
    Dispatch the main window based on application type.

    Parameters
    ----------
    app_type : str
        'sg' for Signal Generator, and
        'rtma' for Real-Time Modal Analyzer.

    Returns
    -------
    QMainWindow
        The main application window instance.
    """
    log.debug(f"Launching app: {app_type.upper()}")
    if app_type == "rtma":
        return RTMAMainWindow()
    
    return SGMainWindow()