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


async def _start_processing_handler(filename: str) -> StreamingResponse:
	"""Common handler for processing requests (supports both GET and POST)"""
	file_path = UPLOAD_DIR / filename
	if not file_path.exists():
		raise HTTPException(status_code=404, detail=f"File not found: {filename}")

	logger.info(f"Starting processing for file: {filename}")
	orchestrator = PipelineOrchestrator(file_path)
	
	async def event_stream():
		async for event in orchestrator.run_pipeline():
			yield event

	return StreamingResponse(
		stream_progress(event_stream()),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"Access-Control-Allow-Origin": "*",
		},
	)


@router.get("/start/{filename}")
async def start_processing_get(filename: str) -> StreamingResponse:
	"""Start processing (GET) - for EventSource/SSE compatibility"""
	return await _start_processing_handler(filename)


@router.post("/start/{filename}")
async def start_processing_post(filename: str) -> StreamingResponse:
	"""Start processing (POST) - for API clients"""
	return await _start_processing_handler(filename)
