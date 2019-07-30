import logging
import os
import sys
import cProfile
import io
import pstats
import re


def init_logger(level, **kwargs):
    """
    initializes the logger
    :param level:       level of the logger, e.g. logging.DEBUG
    :param kwargs:      file_name: name of the file to which should be logged
    :return:
    """

    # initialize the logger
    logger = logging.getLogger()  # configure the root logger
    logger.setLevel(level)

    # create the file handler
    if 'file_name' in kwargs:
        # create the log file path if needed
        file_name = kwargs.get('file_name')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # create the handler
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(level)

        # add the formatter
        formatter_file = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)-15s(%(lineno)-4d) - [%(levelname)-7s] - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    # create the console logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter_console = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)-15s(%(lineno)-4d) - [%(levelname)-7s] - %(message)s', datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter_console)
    logger.addHandler(console_handler)

    logger.debug('logger successfully initialized')


def profile(fnc):
    """
    decorator that can be used to profile function, the output is written to the file profile.txt
    :param fnc:     function for the decorator
    :return:
    """

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        # write the output to a file
        file = open("profile.txt", "w")
        file.write(s.getvalue())
        file.close()

        return retval

    return inner


def bit_not(n, bit_length):
    """
    defines the logical not operation
    :param n:           the number to which the not operation is applied
    :param bit_length:   the length of the bit to apply the not operation
    :return:
    """
    return (1 << bit_length) - 1 - n


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    sort function for natural sort, this will sort strings like this
    "something1", "something12", "something17", "something2", "something25"
    use it like this:  list.sort(key=natural_keys)
    :param text:
    :return:
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]
