import logging
from datetime import datetime
import os
import sys

class FlagLogger:
    """A wrapper to allow logger.info(flag, message) syntax."""
    def __init__(self, logger):
        self.logger = logger

    def info(self, flag, msg, *args, **kwargs):
        self.logger.info(msg, *args, extra={"flag": flag}, **kwargs)

    def warning(self, flag, msg, *args, **kwargs):
        self.logger.warning(msg, *args, extra={"flag": flag}, **kwargs)

    def error(self, flag, msg, *args, **kwargs):
        self.logger.error(msg, *args, extra={"flag": flag}, **kwargs)

    def debug(self, flag, msg, *args, **kwargs):
        self.logger.debug(msg, *args, extra={"flag": flag}, **kwargs)

class DynamicStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that dynamically looks up sys.stdout or sys.stderr
    at the time of log emission, adapting perfectly to context-manager overrides.
    """
    def __init__(self, stream_attr="stdout"):
        super().__init__()
        self._stream_attr = stream_attr

    @property
    def stream(self):
        return getattr(sys, self._stream_attr)

    @stream.setter
    def stream(self, value):
        # Absorb static assignments from the parent constructor
        pass

# ====================== INITIALIZE LOGGER ======================
def setup_logger(log_dir, log_name, logger_obj_name="logger_obj_name"):
    """
    Configure the logger system: output to both console and file simultaneously.
    """
    # 1. If the log directory doesn't exist, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    # 💡 FIX: If the logger has already been configured by another module, 
    # don't wipe it out! Just wrap it and return it.
    logger = logging.getLogger(logger_obj_name)
    if logger.handlers:
        logger = FlagLogger(logger)
        return logger

    # 2. Generate a timestamped log filename, e.g., scraper_log_20231027_103000.log
    log_filename = f"{log_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 3. Create Logger object
    logger.setLevel(logging.INFO) # Set the minimum logger level

    # --- Define a unified format (time accurate to the second) ---
    # %(asctime)s : Time
    # %(levelname)s : Log level (INFO/ERROR)
    # %(message)s : Your message content
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s][%(flag)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s][%(flag)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )

    # --- Handler 1: File output (detailed, with timestamp) ---
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- 💡 FIX: Handler 2: Console output (dynamically follows sys.stdout) ---
    console_handler = DynamicStreamHandler("stdout")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- Return the Wrapped Logger ---
    logger = FlagLogger(logger)
    logger.info("SETUP LOG", f"✅ logger System Started")
    logger.info("SETUP LOG", f"Log file path: {log_filepath}")
    return logger