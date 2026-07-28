"""Define log for whole PEARL program"""


import logging
import logging.handlers


# ========================================================================
# Define color
# ========================================================================
class CustomFormatter(logging.Formatter):
    """
    Custom Formatter
    """

    mode_debug = True

    # --------------------------------------------------------------------
    # Define color
    # --------------------------------------------------------------------
    black = "\x1b[30m"

    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    italics_red = "\x1b[31;3m"

    green = "\x1b[32m"
    bold_green = "\x1b[32;1m"
    italics_green = "\x1b[32;3m"
    light_green = "\x1b[92m"
    bold_light_green = "\x1b[92;1m"

    yellow = "\x1b[33m"
    bold_yellow = "\x1b[33;1m"
    italics_yellow = "\x1b[33;3m"

    blue = "\x1b[34m"
    bold_blue = "\x1b[34;1m"
    italics_blue = "\x1b[34;3m"
    light_blue = "\x1b[94m"
    bold_light_blue = "\x1b[94;1;3m"

    magenta = "\x1b[35m"
    bold_magenta = "\x1b[95;1m"
    italics_magenta = "\x1b[35;3m"

    cyan = "\x1b[36m"
    bold_cyan = "\x1b[36;1m"
    italics_cyan = "\x1b[36;3m"

    white = "\x1b[37m"
    bold_white = "\x1b[37;1m"
    italics_white = "\x1b[37;3m"

    reset = "\x1b[0m"

    # --------------------------------------------------------------------
    # Define message output
    # --------------------------------------------------------------------
    if mode_debug:
        _format = "Process:%(process)d||%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]: %(message)s" # pylint: disable=line-too-long
        _init = "Process:%(process)d||%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]:" # pylint: disable=line-too-long
        _message = "%(message)s"
    else:
        _format = "Process:%(process)d||%(asctime)s-%(levelname)s: %(message)s"
        _init = "Process:%(process)d||%(asctime)s-%(levelname)s:"
        _message = "%(message)s"

    # --------------------------------------------------------------------
    # Format each level
    # --------------------------------------------------------------------
    FORMATS = {
        logging.DEBUG: bold_green + _init + reset + bold_light_blue + _message + reset,
        logging.INFO: bold_cyan + _init + reset + italics_blue + _message + reset,
        logging.WARNING: bold_yellow + _init + reset + italics_yellow + _message + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset,
    }

    # --------------------------------------------------------------------
    # Function apply log
    # --------------------------------------------------------------------
    def format(self, record):
        """Log apply each record"""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# ========================================================================
# Set up global logger object
# ========================================================================
def set_up_ai_logger(name="pearl", save_log_dir="./pearl.log"):
    """Set up global logger object"""
    # --------------------------------------------------------------------
    # Create new logger
    # --------------------------------------------------------------------
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # --------------------------------------------------------------------
    # Set up handler
    # --------------------------------------------------------------------
    # file_handler = logging.handlers.RotatingFileHandler(
    #                                                         save_log_dir,
    #                                                         maxBytes=10485760,
    #                                                         backupCount=300,
    #                                                         encoding="utf-8",
    #                                                     )
    # file_handler.setFormatter(CustomFormatter())
    # logger.addHandler(file_handler)

    # --------------------------------------------------------------------
    # For console
    # --------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # --------------------------------------------------------------------
    # Add new console_handler and turn off propagate
    # https://stackoverflow.com/questions/19561058/duplicate-output-in-simple-python-logging-configuration/19561320#19561320
    # --------------------------------------------------------------------
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    return logger


aiLogger = set_up_ai_logger()
