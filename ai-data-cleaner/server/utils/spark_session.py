from functools import lru_cache
from typing import Optional, Dict, Any
import threading
from contextlib import contextmanager

from pyspark.sql import SparkSession

try:
    from .logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)


class SparkSessionManager:
    """Manages Spark session lifecycle with proper cleanup and resource tracking."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._session = None
            self._session_config = {}
            self._reference_count = 0
            self._initialized = True
            logger.info("SparkSessionManager initialized")
    
    def get_session(self, app_name: str = "AIDataCleaner", master: Optional[str] = None, 
                   config: Optional[Dict[str, str]] = None) -> SparkSession:
        """Get or create Spark session with reference counting."""
        with self._lock:
            if self._session is None:
                self._create_session(app_name, master, config)
            self._reference_count += 1
            logger.info(f"Spark session requested. Reference count: {self._reference_count}")
            return self._session
    
    def _create_session(self, app_name: str, master: Optional[str], config: Optional[Dict[str, str]]):
        """Create new Spark session with optimized configuration."""
        builder = SparkSession.builder.appName(app_name)
        
        if master:
            builder = builder.master(master)
        
        # Optimized configuration for data processing
        default_config = {
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.execution.arrow.maxRecordsPerBatch": "10000"
        }
        
        # Merge with user config
        final_config = {**default_config, **(config or {})}
        for key, value in final_config.items():
            builder = builder.config(key, value)
        
        self._session = builder.getOrCreate()
        self._session_config = final_config
        logger.info(f"Created Spark session '{app_name}' with config: {final_config}")
    
    def release_session(self):
        """Release reference to Spark session."""
        with self._lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.info(f"Spark session released. Reference count: {self._reference_count}")
                
                if self._reference_count == 0:
                    self._cleanup_session()
    
    def _cleanup_session(self):
        """Clean up Spark session when no longer needed."""
        if self._session is not None:
            try:
                logger.info("Cleaning up Spark session...")
                self._session.stop()
                self._session = None
                self._session_config = {}
                logger.info("Spark session cleaned up successfully")
            except Exception as e:
                logger.error(f"Failed to cleanup Spark session: {e}")
                raise RuntimeError(f"Failed to cleanup Spark session: {e}")
    
    def force_cleanup(self):
        """Force cleanup of Spark session regardless of reference count."""
        with self._lock:
            self._reference_count = 0
            self._cleanup_session()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about current session."""
        return {
            "has_session": self._session is not None,
            "reference_count": self._reference_count,
            "config": self._session_config.copy() if self._session_config else {}
        }


# Global session manager instance
_session_manager = SparkSessionManager()


@contextmanager
def spark_session_context(app_name: str = "AIDataCleaner", master: Optional[str] = None,
                         config: Optional[Dict[str, str]] = None):
    """Context manager for Spark session with automatic cleanup."""
    session = None
    try:
        session = _session_manager.get_session(app_name, master, config)
        yield session
    finally:
        if session:
            _session_manager.release_session()


def get_spark(app_name: str = "AIDataCleaner", master: Optional[str] = None, 
              config: Optional[Dict[str, str]] = None) -> SparkSession:
    """Get Spark session with proper lifecycle management."""
    return _session_manager.get_session(app_name, master, config)


def release_spark():
    """Release reference to Spark session."""
    _session_manager.release_session()


def stop_spark():
    """Force stop Spark session and cleanup resources."""
    _session_manager.force_cleanup()


def get_spark_info() -> Dict[str, Any]:
    """Get information about current Spark session."""
    return _session_manager.get_session_info()


# Legacy function for backward compatibility
@lru_cache(maxsize=1)
def get_spark_legacy(app_name: str = "AIDataCleaner", master: Optional[str] = None) -> SparkSession:
    """Legacy function - use get_spark() instead for proper lifecycle management."""
    logger.warning("Using legacy get_spark_legacy() - consider using get_spark() for better resource management")
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    return builder.getOrCreate()


