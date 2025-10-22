from pathlib import Path
from typing import Literal, Optional

import pandas as pd

try:
    from ..config import ALLOWED_EXTENSIONS, UPLOAD_DIR, PROCESSED_DIR
    from .logger import get_logger
except ImportError:
    from config import ALLOWED_EXTENSIONS, UPLOAD_DIR, PROCESSED_DIR
    from utils.logger import get_logger


FileKind = Literal["csv", "json", "parquet"]
logger = get_logger(__name__)


def detect_file_kind(path: Path) -> FileKind:
	if path.suffix.lower() not in ALLOWED_EXTENSIONS:
		raise ValueError(f"Unsupported extension: {path.suffix}")
	if path.suffix.lower() == ".csv":
		return "csv"
	if path.suffix.lower() == ".json":
		return "json"
	return "parquet"


def save_upload(file_bytes: bytes, filename: str) -> Path:
	UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
	path = (UPLOAD_DIR / filename).resolve()
	path.write_bytes(file_bytes)
	logger.info(f"Saved upload to {path}")
	return path


def read_pandas(path: Path, kind: Optional[FileKind] = None) -> pd.DataFrame:
	kind = kind or detect_file_kind(path)
	if kind == "csv":
		return pd.read_csv(path)
	if kind == "json":
		return pd.read_json(path, lines=False)
	return pd.read_parquet(path)


def write_pandas(df: pd.DataFrame, out_name: str, kind: Optional[FileKind] = None) -> Path:
	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
	outfile = (PROCESSED_DIR / out_name).resolve()
	kind = kind or detect_file_kind(outfile)
	if kind == "csv":
		df.to_csv(outfile, index=False)
	elif kind == "json":
		df.to_json(outfile, orient="records")
	else:
		df.to_parquet(outfile, index=False)
	logger.info(f"Wrote processed dataset to {outfile}")
	return outfile


