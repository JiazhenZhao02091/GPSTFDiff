import logging
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *args, **kwargs):
        return str(text)


class FilterDuplicateWarning(logging.Filter):

    def __init__(self, name: str = 'fusion'):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class ColorfulFormatter(logging.Formatter):
    _color_mapping: dict = dict(
        ERROR='red', WARNING='yellow', INFO='white', DEBUG='green'
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), 'blink should only be available when color is True'

        error_prefix = self._get_prefix('ERROR', color, blink=True)
        warn_prefix = self._get_prefix('WARNING', color, blink=True)
        info_prefix = self._get_prefix('INFO', color, blink)
        debug_prefix = self._get_prefix('DEBUG', color, blink)


        self.err_format = f'%(asctime)s - %(name)s - {error_prefix} - %(pathname)s - %(funcName)s - %(lineno)d - %(message)s'
        self.warn_format = f'%(asctime)s - %(name)s - {warn_prefix} - %(message)s'
        self.info_format = f'%(asctime)s - %(name)s - {info_prefix} - %(message)s'
        self.debug_format = f'%(asctime)s - %(name)s - {debug_prefix} - %(message)s'

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        if color:
            attrs = ['underline']
            if blink:
                attrs.append('blink')
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class FusionLogger(Logger):

    def __init__(
        self,
        logger_name='fusion',
        log_file: Optional[str] = None,
        log_level: Union[int, str] = 'INFO',
        file_mode: str = 'w',
    ):
        super(FusionLogger, self).__init__(logger_name)

        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]


        stream_handler = logging.StreamHandler(stream=sys.stdout)

        stream_handler.setFormatter(
            ColorfulFormatter(color=True, datefmt='%m/%d %H:%M:%S')
        )

        stream_handler.setLevel(log_level)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:


            file_handler = logging.FileHandler(log_file, file_mode)


            file_handler.setFormatter(
                ColorfulFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S')
            )
            file_handler.setLevel(log_level)
            file_handler.addFilter(FilterDuplicateWarning(logger_name))
            self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file


    def setLevel(self, level):
        self.level = logging._checkLevel(level)


def get_logger(
    logger_name='fusion',
    log_file: Optional[str] = None,
    log_level: Union[int, str] = 'INFO',
    file_mode: str = 'w',
):
    logger = logging.getLogger(logger_name)
    if isinstance(log_level, str):
        log_level = logging._nameToLevel[log_level]


    stream_handler = logging.StreamHandler(stream=sys.stdout)

    stream_handler.setFormatter(ColorfulFormatter(color=True, datefmt='%m/%d %H:%M:%S'))

    stream_handler.setLevel(log_level)
    stream_handler.addFilter(FilterDuplicateWarning(logger_name))
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(
            ColorfulFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S')
        )
        file_handler.setLevel(log_level)
        file_handler.addFilter(FilterDuplicateWarning(logger_name))
        logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':


    txt_loger = FusionLogger(
            logger_name="ST",
            log_file= './log.log',
            log_level="INFO",
        )

    txt_loger.info('test_info')
