import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from ..config import LOG_DIR
except ImportError:
    from config import LOG_DIR


def get_logger(name: str = "ai_data_cleaner", log_file: Optional[Path] = None) -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger
	logger.setLevel(logging.INFO)

	LOG_DIR.mkdir(parents=True, exist_ok=True)
	log_path = log_file or (LOG_DIR / f"{name}.log")

	file_handler = RotatingFileHandler(str(log_path), maxBytes=10 * 1024 * 1024, backupCount=5)
	file_handler.setLevel(logging.INFO)
	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	logger.propagate = False
	return logger


