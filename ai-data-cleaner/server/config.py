import os
from pathlib import Path


# Core configuration for server paths and thresholds
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / ".."

# Directories
UPLOAD_DIR = (PROJECT_ROOT / "uploads").resolve()
SNAPSHOT_DIR = (PROJECT_ROOT / "snapshots").resolve()
LOG_DIR = (PROJECT_ROOT / "logs").resolve()
PROCESSED_DIR = (PROJECT_ROOT / "processed_data").resolve()
CLIENT_DIR = (PROJECT_ROOT / "client").resolve()

# Thresholds and defaults
FILE_SIZE_SPARK_THRESHOLD_MB = int(os.getenv("FILE_SIZE_SPARK_THRESHOLD_MB", "15"))
SAMPLE_ROWS_PER_COLUMN = int(os.getenv("SAMPLE_ROWS_PER_COLUMN", "20"))
MAX_SAMPLE_ROWS = int(os.getenv("MAX_SAMPLE_ROWS", "20"))
MAX_UNIQUE_PREVIEW = int(os.getenv("MAX_UNIQUE_PREVIEW", "50"))

# Gemini / Google Generative AI
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Security / Upload
ALLOWED_EXTENSIONS = {".csv", ".json", ".parquet"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))  # 1 GB cap


def ensure_directories() -> None:
	"""Create required directories if missing."""
	for path in [UPLOAD_DIR, SNAPSHOT_DIR, LOG_DIR, PROCESSED_DIR]:
		path.mkdir(parents=True, exist_ok=True)


def is_big_data(file_path: Path) -> bool:
	"""Return True if file exceeds the Spark threshold in MB."""
	size_mb = file_path.stat().st_size / (1024 * 1024)
	return size_mb >= FILE_SIZE_SPARK_THRESHOLD_MB


def is_extension_allowed(file_path: Path) -> bool:
	return file_path.suffix.lower() in ALLOWED_EXTENSIONS


