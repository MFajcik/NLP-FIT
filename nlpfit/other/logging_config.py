# PyVersion: 3.6
# Authors: Martin Fajčík, Viliam Samuel Hošťák
import os
import sys
import datetime
import logging
import logging.config
import yaml

from nlpfit.other.util import touch, Deprecated


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    fpath = os.path.join(logpath, module, timestamp)

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, timestamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        lpath=os.path.join(logpath, timestamp)
        if not os.path.exists(lpath):
           os.makedirs(lpath)
        logging.basicConfig(level=default_level, filename=os.path.join(lpath,"base.log"))


@Deprecated
def init_logging(module, logpath=os.getcwd()):
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
    path = os.path.join(logpath, "log/")
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
    """
    Duck-typed logger-like object
    """

    def info(self, str):
        print(str)

    def debug(self, str):
        self.info(str)

    def error(self, str):
        sys.stderr.write(str)
        sys.stderr.write("\n")

    def critical(self, str):
        self.error(str)
