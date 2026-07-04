import logging

from .logger_factory import LoggerFactory
from ..config.settings import settings
from textwrap import fill

import numpy as np
import pandas as pd

class Logger:
    def __init__(self):
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = LoggerFactory.create(
                settings.logging
            )
            self.logger.info("RAG System started")
        else:
            self.logger.info("RAG System already started")

    def _get_logger(self):
        return self.logger

    # def _get_logger(self):
    #     return logging.getLogger(__name__)

        
    def _log(self, level, message, width=80):
        if isinstance(message, str):
            message = fill(message, width=width)
            self.logger.log(level, message)
        
        elif isinstance(message, (pd.DataFrame, np.ndarray)):
            check = message.shape if isinstance(message, np.ndarray) else message.shape
            if check[0] > 10:
                message = message.head(5)
            self.logger.log(level, message.to_string())
        
        else:
            self.logger.log(level, message)
    
    # def _log(self, level, message):
    #     logger = self._get_logger()
    #     logger.log(level, message)
    #     print(f"[{logging.getLevelName(level)}] {message}")
        
        
    def error(self, message, width=80):
        self._log(logging.ERROR, message, width)
        
        
    def warning(self, message, width=80):
        self._log(logging.WARNING, message, width)
        
    def info(self, message, width=80):
        self._log(logging.INFO, message, width)
        
    def debug(self, message, width=80):
        self._log(logging.DEBUG, message, width)






#         import logging

# class Logger:
#     def __init__(self):
#         pass

#     def _get_logger(self):
#         return logging.getLogger(__name__)

#     def _log(self, level, message):
#         logger = self._get_logger()
#         logger.log(level, message)
#         print(f"[{logging.getLevelName(level)}] {message}")
        
#     def error(self, message):
#         self._log(logging.ERROR, message)
        
#     def warning(self, message):
#         self._log(logging.WARNING, message)
        
#     def info(self, message):
#         self._log(logging.INFO, message)
        
#     def debug(self, message):
#         self._log(logging.DEBUG, message)