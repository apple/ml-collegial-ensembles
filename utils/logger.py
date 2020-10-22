import logging
import sys

logger_level = logging.INFO  # Change this to NOTSET, DEBUG, INFO, WARNING, ERROR or CRITICAL depending on the needs.

ESC = '\x1b'
RESET_SEQ = ESC + '[0m'
COLOR_SEQ = ESC + '[0;%dm'
BOLD_SEQ =  ESC + '[1m'
BOLD_COLOR_SEQ = ESC + '[1;%dm'
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
LEVEL_COLORS = {
    'DEBUG': BLUE,
    'INFO': GREEN,
    'WARNING': YELLOW,
    'ERROR': RED,
    'CRITICAL': MAGENTA,
}


def bold_color_line(color, text):
    return BOLD_COLOR_SEQ % color + text + RESET_SEQ


def color_line(color, text):
    return COLOR_SEQ % color + text + RESET_SEQ


class LessThenFilter(logging.Filter):
    """
    Less than logging filter for logging everything less than a given logger level. This is useful e.g. to
    separate the critical logger output (e.g. from logger.ERROR) from the normal running output (e.g. logger.INFO).
    """
    def __init__(self, level):
        self._level = level
        logging.Filter.__init__(self)

    def filter(self, rec):
        return rec.levelno < self._level


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        level_name = record.levelname
        level_color = LEVEL_COLORS.get(level_name)
        if level_color:
            record.levelname = bold_color_line(level_color, level_name)
            record.msg = color_line(level_color, record.msg)
        return logging.Formatter.format(self, record)


formatter = ColoredFormatter('%(asctime)s '
                             '(%(levelname)s) '
                             '%(funcName)s '
                             '(%(filename)s '
                             '%(lineno)d):\t'
                             '%(message)s')

# Create stdout handler - only logs messages less than warning
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(LessThenFilter(logging.WARNING))
# Create stderr handler - only logs messages greater than or equal to warning
stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(formatter)

# Create logger
logger = logging.getLogger('collegial-ensemble')
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)
logger.setLevel(logger_level)


def add_log_file(logger_obj, logfile, logger_level):
    logger_obj.info('Adding logfile "{}" to logger "{}"'.format(logfile, logger_obj.name))
    # Create logfile handler - logs all messages
    logfile_handler = logging.FileHandler(logfile)
    logfile_handler.setLevel(logger_level)
    logfile_handler.setFormatter(formatter)
    logger_obj.addHandler(logfile_handler)
