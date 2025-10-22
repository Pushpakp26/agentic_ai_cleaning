import pandas as pd
import numpy as np

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger


logger = get_logger(__name__)


class NormalizeAgent(BaseAgent):
	"""Agent for normalizing numeric columns to [0, 1] range."""
	
	def __init__(self):
		super().__init__("NormalizeAgent")
	
	def process(self, df: pd.DataFrame, column: str = None, method: str = "minmax", **kwargs) -> pd.DataFrame:
		"""Normalize numeric columns to [0, 1] range."""
		df = df.copy()
		
		if column:
			if pd.api.types.is_numeric_dtype(df[column]):
				df = self._normalize_column(df, column, method)
		else:
			# Normalize all numeric columns
			numeric_cols = df.select_dtypes(include=[np.number]).columns
			for col in numeric_cols:
				df = self._normalize_column(df, col, method)
		
		logger.info(f"Normalization completed using method: {method}")
		return df
	
	def _normalize_column(self, df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
		"""Normalize a specific numeric column."""
		series = df[column]
		
		if method == "minmax":
			min_val, max_val = series.min(), series.max()
			if max_val != min_val:
				df[column] = (series - min_val) / (max_val - min_val)
			else:
				df[column] = 0  # Handle case where all values are the same
		
		elif method == "robust":
			# Use 25th and 75th percentiles for robustness to outliers
			q25, q75 = series.quantile(0.25), series.quantile(0.75)
			if q75 != q25:
				df[column] = (series - q25) / (q75 - q25)
			else:
				df[column] = 0
		
		logger.info(f"Normalized column '{column}' using {method} method")
		return df
