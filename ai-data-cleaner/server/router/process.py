from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

try:
    from ..agents.orchestrator import PipelineOrchestrator
    from ..config import UPLOAD_DIR
    from ..sse import stream_progress
    from ..utils.logger import get_logger
except ImportError:
    from agents.orchestrator import PipelineOrchestrator
    from config import UPLOAD_DIR
    from sse import stream_progress
    from utils.logger import get_logger


router = APIRouter(prefix="/process", tags=["process"])
logger = get_logger(__name__)


@router.get("/start/{filename}")
async def start_processing_get(filename: str):
    """Start processing (GET) - for EventSource/SSE compatibility"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    logger.info(f"Starting processing for file: {filename}")
    orchestrator = PipelineOrchestrator(file_path)

    return StreamingResponse(
        stream_progress(orchestrator.run_pipeline()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/start/{filename}")
async def start_processing_post(filename: str):
    """Start processing (POST) - for API clients"""
    # (Optional) reuse the same logic for POST if needed
    return await start_processing_get(filename)
