import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    # Try relative imports first (when run as module)
    from .config import ensure_directories, CLIENT_DIR
    from .utils.logger import get_logger
    from .utils.spark_session import get_spark, stop_spark
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import ensure_directories, CLIENT_DIR
    from utils.logger import get_logger
    from utils.spark_session import get_spark, stop_spark


logger = get_logger()
app = FastAPI(title="AI Data Cleaner", version="0.1.0")


# CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    """Initialize required directories and start Spark session."""
    ensure_directories()
    logger.info("Server startup: directories ensured.")

    try:
        spark = get_spark()
        logger.info("[OK] Spark session started successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start Spark session: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanly stop the Spark session when the server shuts down."""
    try:
        stop_spark()
        logger.info("[CLEANUP] Spark session stopped and cache cleared successfully.")
    except Exception as e:
        logger.warning(f"Error while stopping Spark session: {e}")


@app.get("/health")
async def health_check():
	"""Health check endpoint to verify server is running."""
	return {
		"status": "healthy",
		"service": "AI Data Cleaner",
		"version": "0.1.0"
	}


@app.get("/api/health")
async def api_health_check():
	"""API health check endpoint."""
	return {
		"status": "healthy",
		"service": "AI Data Cleaner API",
		"version": "0.1.0",
		"endpoints": {
			"upload": "/api/upload/",
			"process": "/api/process/start/{filename}",
			"download_dataset": "/api/download/dataset/{filename}",
			"download_report": "/api/download/report/{filename}",
			"list_uploads": "/api/upload/list",
			"list_processed": "/api/download/list"
		}
	}


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
	"""Suppress Chrome DevTools 404 - return empty config."""
	return {}


# Routers will be included after implementation
try:
	# Try relative imports first (when run as module)
	from .router.upload import router as upload_router
	from .router.process import router as process_router
	from .router.download import router as download_router
	logger.info("Successfully imported routers with relative imports")
except ImportError as e:
	logger.warning(f"Relative imports failed: {e}, trying absolute imports")
	# Fall back to absolute imports (when run directly from server directory)
	from router.upload import router as upload_router
	from router.process import router as process_router
	from router.download import router as download_router
	logger.info("Successfully imported routers with absolute imports")

try:
	app.include_router(upload_router, prefix="/api")
	app.include_router(process_router, prefix="/api")
	app.include_router(download_router, prefix="/api")
	logger.info("All routers mounted successfully")
except Exception as e:
	logger.error(f"Failed to mount routers: {e}")
	import traceback
	logger.error(f"Traceback: {traceback.format_exc()}")


# Mount client as static (optional local preview) - MUST be last
# Note: This catches all remaining routes, so API routes must be registered first
if CLIENT_DIR.exists():
	try:
		app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")
		logger.info(f"Mounted static client files from {CLIENT_DIR}")
	except Exception as e:
		logger.warning(f"Failed to mount static files: {e}")


def run():
	uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
	run()


