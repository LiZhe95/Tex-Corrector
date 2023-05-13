# -*- coding: utf-8 -*-
import logging
import logging.handlers
import os


def get_logger(name, log_file=None, log_level='DEBUG'):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :param log_level: 日志级别
    :return:
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level.upper())
    formatter = logging.Formatter('[%(levelname)7s %(asctime)s %(module)s:%(lineno)4d] %(message)s',
                                  datefmt='%Y%m%d %H:%M:%S')
    if log_file:
        dirname = os.path.dirname(log_file)
        os.makedirs(dirname, exist_ok=True)
        # f_handle = logging.FileHandler(log_file)
        f_handle = logging.handlers.TimedRotatingFileHandler(filename=log_file, when="D", interval=1, backupCount=5, encoding='utf8')
        f_handle.setFormatter(formatter)
        logger.addHandler(f_handle)
    handle = logging.StreamHandler()
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    return logger


