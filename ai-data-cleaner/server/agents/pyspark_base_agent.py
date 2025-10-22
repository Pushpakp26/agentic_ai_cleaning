# base/pyspark_base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pyspark.sql import DataFrame
import pandas as pd

try:
    from .inspector_interface import ProcessingAgentInterface
    from ..utils.logger import get_logger
except ImportError:
    from agents.inspector_interface import ProcessingAgentInterface
    from utils.logger import get_logger

logger = get_logger(__name__)


class PySparkBaseAgent(ProcessingAgentInterface):
    """Abstract base class for all PySpark agents that transform DataFrames."""

    def __init__(self, spark=None, df=None, name: str = None):
        """
        Initialize PySpark agent.
        
        Args:
            spark: SparkSession instance (optional, for compatibility)
            df: DataFrame (optional, deprecated - pass df to process() instead)
            name: Agent name (optional, defaults to class name)
        """
        super().__init__(name or self.__class__.__name__)
        self.spark = spark
        self.df = df  # Kept for backward compatibility but not recommended
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Process the DataFrame and return the result.
        
        Args:
            df: Input DataFrame (PySpark or Pandas - will be converted if needed)
            **kwargs: Additional parameters for processing
            
        Returns:
            Processed DataFrame of the same type as input
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this agent's processing capabilities."""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "agent_class": self.__class__.__name__,
            "supports_pandas": False,  # PySpark agents primarily work with Spark DataFrames
            "supports_pyspark": True,
            "has_spark_session": self.spark is not None
        })
        return base_metadata
    
    def validate_input(self, df: Union[DataFrame, pd.DataFrame]) -> bool:
        """Validate input DataFrame type without triggering Spark actions."""
        if df is None:
            raise ValueError("Input DataFrame cannot be None")
        
        if isinstance(df, DataFrame):
            # Avoid df.count(); assume non-empty; downstream ops will fail if invalid
            return True
        elif isinstance(df, pd.DataFrame):
            # Pandas empty check is cheap
            if df.empty:
                logger.warning("Input Pandas DataFrame is empty")
            return True
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")
    
    def log_processing_start(self, column: Optional[str] = None, **kwargs):
        """Log the start of processing operation."""
        context = f"column '{column}'" if column else "entire dataset"
        logger.info(f"[{self.name}] Starting PySpark processing on {context}")
        if kwargs:
            logger.debug(f"[{self.name}] Parameters: {kwargs}")
    
    def log_processing_end(self, input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None, column: Optional[str] = None):
        """Log completion without forcing actions like count()."""
        context = f"column '{column}'" if column else "dataset"
        if input_shape is not None and output_shape is not None:
            logger.info(f"[{self.name}] Completed PySpark processing {context}: {input_shape} -> {output_shape}")
        else:
            logger.info(f"[{self.name}] Completed PySpark processing {context}")
