from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    from ..config import MAX_UPLOAD_MB, ALLOWED_EXTENSIONS, UPLOAD_DIR, is_extension_allowed
    from ..utils.file_handler import save_upload
    from ..utils.logger import get_logger
except ImportError:
    from config import MAX_UPLOAD_MB, ALLOWED_EXTENSIONS, UPLOAD_DIR, is_extension_allowed
    from utils.file_handler import save_upload
    from utils.logger import get_logger


router = APIRouter(prefix="/upload", tags=["upload"])
logger = get_logger(__name__)


@router.get("/list")
async def list_uploaded_files():
    """List all uploaded files."""
    try:
        if not UPLOAD_DIR.exists():
            return JSONResponse({"files": [], "message": "No uploaded files yet"})
        
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "uploaded": file_path.stat().st_mtime,
                    "extension": file_path.suffix
                })
        
        logger.info(f"Listed {len(files)} uploaded files")
        return JSONResponse({"files": files, "count": len(files)})
    except Exception as e:
        logger.error(f"Failed to list uploaded files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.post("/")
async def upload_file(file: UploadFile):
    """Upload a data file (CSV, JSON, or Parquet)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    path = Path(file.filename)
    if not is_extension_allowed(path):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {list(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_UPLOAD_MB} MB, got: {size_mb:.2f} MB",
        )

    saved_path = save_upload(content, file.filename)
    logger.info(f"Uploaded {file.filename} -> {saved_path} ({size_mb:.2f} MB)")

    return JSONResponse({
        "filename": file.filename,
        "path": str(saved_path),
        "size_mb": size_mb,
        "message": "File uploaded successfully",
    })
