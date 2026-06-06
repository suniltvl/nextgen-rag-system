import logging

from .logger_factory import LoggerFactory
from ..config.settings import settings

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

        
    def _log(self, level, message):
        self.logger.log(level, message)
    
    # def _log(self, level, message):
    #     logger = self._get_logger()
    #     logger.log(level, message)
    #     print(f"[{logging.getLevelName(level)}] {message}")
        
        
    def error(self, message):
        self.logger.error(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def info(self, message):
        self.logger.info(message)
        
    def debug(self, message):
        self.logger.debug(message)






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