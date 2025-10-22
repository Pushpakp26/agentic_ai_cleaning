from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

try:
    from ..config import PROCESSED_DIR
    from ..utils.logger import get_logger
except ImportError:
    from config import PROCESSED_DIR
    from utils.logger import get_logger


router = APIRouter(prefix="/download", tags=["download"])
logger = get_logger(__name__)


@router.get("/list")
async def list_processed_files() -> JSONResponse:
	"""List all available processed files and reports."""
	try:
		if not PROCESSED_DIR.exists():
			return JSONResponse({"files": [], "message": "No processed files yet"})
		
		files = []
		for file_path in PROCESSED_DIR.iterdir():
			if file_path.is_file():
				files.append({
					"filename": file_path.name,
					"size_bytes": file_path.stat().st_size,
					"size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
					"modified": file_path.stat().st_mtime,
					"type": "report" if file_path.suffix in [".html", ".pdf"] else "dataset"
				})
		
		logger.info(f"Listed {len(files)} processed files")
		return JSONResponse({"files": files, "count": len(files)})
	except Exception as e:
		logger.error(f"Failed to list files: {e}")
		raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/dataset/{filename}")
async def download_dataset(filename: str) -> FileResponse:
	"""Download a processed dataset file."""
	file_path = PROCESSED_DIR / filename
	if not file_path.exists():
		raise HTTPException(status_code=404, detail=f"Processed dataset not found: {filename}")

	logger.info(f"Serving download: {filename}")
	return FileResponse(
		path=str(file_path),
		filename=filename,
		media_type="application/octet-stream",
	)


@router.get("/report/{filename}")
async def download_report(filename: str) -> FileResponse:
	"""Download a report file (HTML or PDF)."""
	file_path = PROCESSED_DIR / filename
	if not file_path.exists():
		raise HTTPException(status_code=404, detail=f"Report not found: {filename}")

	media_type = "text/html" if filename.endswith(".html") else "application/pdf"
	logger.info(f"Serving report download: {filename}")
	return FileResponse(
		path=str(file_path),
		filename=filename,
		media_type=media_type,
	)
