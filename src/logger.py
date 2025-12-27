import logging
import sys
import os

class Logger:
    def __init__(self, name: str, log_file: str = None, level: str = 'INFO'):
        """
        Custom Logger.
        
        Args:
            name (str): Name of the logger.
            log_file (str): Path to log file. If None, only logs to console.
            level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_level(level))
        self.logger.propagate = False # Prevent double logging

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s', 
            datefmt='%H:%M:%S'
        )

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if log_file:
                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def _get_level(self, level_str: str) -> int:
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level_str.upper(), logging.INFO)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)