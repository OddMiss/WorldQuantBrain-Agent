import logging
import datetime
import os
import sys

# ====================== INITIALIZE LOGGER ======================
def setup_logger(log_dir, log_name, logger_obj_name="logger_obj_name"):
    """
    Configure the logger system: output to both console and file simultaneously.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    log_filename = f"{log_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(logger_obj_name)
    logger.setLevel(logging.INFO) 
    logger.propagate = False # (Added this to stop duplicate printing)
    logger.handlers = [] 

    file_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt="%y-%#m-%#d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout) # 
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"✅ logger System Started")
    logger.info(f"Log file path: {log_filepath}")
    return logger