import logging
import sys

from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger
from .helper import helper


class LoggerFactory:

    @staticmethod
    def create(config):

        logger = logging.getLogger("real_world_rag")

        if logger.handlers:
            return logger

        logger.setLevel(config.log_level)

        logger_dir_path = Path(helper.get_project_root() / config.log_dir)

        print(f"Logger directory: {logger_dir_path}")

        logger_dir_path.mkdir(
            parents=True,
            exist_ok=True
        )

        #
        # Console Logger
        #
        if config.enable_console_logger:

            console_handler = logging.StreamHandler(sys.stdout)

            console_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s\t| %(name)s | %(message)s"
            )

            console_handler.setFormatter(console_formatter)

            logger.addHandler(console_handler)

        #
        # File Logger
        #
        if config.enable_file_logger:

            file_handler = TimedRotatingFileHandler(
                filename=f"{logger_dir_path}/{config.log_file}",
                when="midnight",
                backupCount=config.log_retention_days,
                encoding="utf-8"
            )

            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s\t| %(name)s | %(message)s"
            )

            file_handler.setFormatter(file_formatter)

            logger.addHandler(file_handler)

        #
        # JSON Logger
        #
        if config.enable_json_logger:

            json_handler = TimedRotatingFileHandler(
                filename=f"{logger_dir_path}/rag_json.log",
                when="midnight",
                backupCount=config.log_retention_days,
                encoding="utf-8"
            )

            json_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )

            json_handler.setFormatter(json_formatter)

            logger.addHandler(json_handler)

        return logger