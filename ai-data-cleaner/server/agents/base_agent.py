from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
from pyspark.sql import DataFrame

try:
    from .inspector_interface import ProcessingAgentInterface
    from ..utils.logger import get_logger
except ImportError:
    from agents.inspector_interface import ProcessingAgentInterface
    from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ProcessingAgentInterface):
	"""Base class for all data processing agents that transform DataFrames."""
	
	def __init__(self, name: str):
		super().__init__(name)
		logger.debug(f"Initialized {self.__class__.__name__}")
	
	@abstractmethod
	def process(self, df: Union[pd.DataFrame, DataFrame], **kwargs) -> Union[pd.DataFrame, DataFrame]:
		"""
		Process the dataframe and return the transformed result.
		
		Args:
			df: Input DataFrame (Pandas or PySpark)
			**kwargs: Additional parameters for processing
			
		Returns:
			Transformed DataFrame of the same type as input
		"""
		pass
	
	def get_metadata(self) -> Dict[str, Any]:
		"""Return metadata about this agent's processing capabilities."""
		base_metadata = super().get_metadata()
		base_metadata.update({
			"agent_class": self.__class__.__name__,
			"supports_pandas": True,
			"supports_pyspark": False  # Override in PySpark agents
		})
		return base_metadata
	
	def validate_input(self, df: Union[pd.DataFrame, DataFrame]) -> bool:
		"""Validate input DataFrame type without triggering heavy Spark actions."""
		if df is None:
			raise ValueError("Input DataFrame cannot be None")
		
		if isinstance(df, pd.DataFrame):
			# Pandas empty check is cheap
			if df.empty:
				logger.warning("Input Pandas DataFrame is empty")
			return True
		elif hasattr(df, 'columns'):
			# Assume valid Spark DataFrame; avoid actions like count()
			return True
		else:
			raise TypeError(f"Unsupported DataFrame type: {type(df)}")
	
	def log_processing_start(self, column: Optional[str] = None, **kwargs):
		"""Log the start of processing operation."""
		context = f"column '{column}'" if column else "entire dataset"
		logger.info(f"[{self.name}] Starting processing on {context}")
		if kwargs:
			logger.debug(f"[{self.name}] Parameters: {kwargs}")
	
	def log_processing_end(self, input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None, column: Optional[str] = None):
		"""Log the completion of processing without requiring expensive shape counts."""
		context = f"column '{column}'" if column else "dataset"
		if input_shape is not None and output_shape is not None:
			logger.info(f"[{self.name}] Completed processing {context}: {input_shape} -> {output_shape}")
		else:
			logger.info(f"[{self.name}] Completed processing {context}")
