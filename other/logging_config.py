# PyVersion: 3.6
# Authors: Martin Fajčík, Viliam Samuel Hošťák
import os
import sys
import datetime
import logging


def init_logging(module,logpath=os.getcwd()):
    """
    Logger initialization
    Parameters:
        module:     Name of module
        logpath:    Path to logging folder [default: script's working directory]
    """
    log_formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = datetime.datetime.now()
    path = os.path.join(logpath,"log/")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file_handler = logging.FileHandler(os.path.join(path, module) + "_" + now.strftime("%Y-%m-%d_%H:%M"))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logging

class logger_stub():
    def info(self, str):
        print(str)
    def critical(self, str):
        sys.stderr.write(str)
        sys.stderr.write("\n")
