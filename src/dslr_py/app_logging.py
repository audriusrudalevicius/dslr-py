import sys
from datetime import datetime
import logging
import os
from logging.handlers import RotatingFileHandler

try:
    import colorlog
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False


def configure_logging(main_level='INFO'):
    log = logging.getLogger()
    map(log.removeHandler, log.handlers[:])
    map(log.removeFilter, log.filters[:])

    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if HAVE_COLORLOG and os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {
            'DEBUG': 'reset',
            'INFO': 'reset',
            'WARNING': 'bold_yellow',
            'ERROR': 'bold_red',
            'CRITICAL': 'bold_red'
        }
        formatter = colorlog.ColoredFormatter(cformat, date_format, log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(main_level)

    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    rt_handler = RotatingFileHandler('app_{}.log'.format(log_filename), maxBytes=100 * 1024 * 1024, backupCount=10)
    rt_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-12s [%(levelname)s] %(message)s')
    rt_handler.setFormatter(formatter)

    logging.basicConfig(
        level=main_level,
        format=format_str,
        handlers=[
            rt_handler,
            stream_handler
        ]
    )

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger('backoff').setLevel(logging.WARNING)
