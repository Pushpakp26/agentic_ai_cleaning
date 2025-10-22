from typing import Dict, Any, Union
import pandas as pd
from pyspark.sql import DataFrame

try:
    from ..inspector_interface import InspectorInterface
    from ...utils.sampling import sample_per_column
    from ...utils.logger import get_logger
except ImportError:
    from agents.inspector_interface import InspectorInterface
    from utils.sampling import sample_per_column
    from utils.logger import get_logger

logger = get_logger(__name__)


class InspectorAgent(InspectorInterface):
	"""Base inspector agent for analyzing data samples."""
	
	def __init__(self):
		super().__init__("InspectorAgent")
		logger.debug("Initialized Pandas InspectorAgent")
	
	def process(self, df: Union[pd.DataFrame, DataFrame], **kwargs) -> Dict[str, Any]:
		"""
		Analyze the dataframe and return inspection results.
		
		Args:
			df: Input DataFrame (Pandas or PySpark)
			**kwargs: Additional parameters for inspection
			
		Returns:
			Dictionary containing inspection results and suggestions
		"""
		if isinstance(df, pd.DataFrame):
			logger.info("Performing Pandas-based inspection")
			return sample_per_column(df)
		else:
			# Convert PySpark DataFrame to Pandas for inspection
			logger.info("Converting PySpark DataFrame to Pandas for inspection")
			pandas_df = df.limit(10000).toPandas()  # Limit for performance
			return sample_per_column(pandas_df)
