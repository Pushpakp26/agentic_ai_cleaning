import pandas as pd
import numpy as np

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger


logger = get_logger(__name__)


class MeanMedianImputerAgent(BaseAgent):
	"""Agent for imputing missing values using mean, median, or mode."""
	
	def __init__(self):
		super().__init__("MeanMedianImputerAgent")
	
	def process(self, df: pd.DataFrame, column: str = None, strategy: str = "auto", **kwargs) -> pd.DataFrame:
		"""Impute missing values in the dataframe."""
		df = df.copy()
		
		if column:
			df = self._impute_column(df, column, strategy)
		else:
			# Impute all columns with missing values
			for col in df.columns:
				if df[col].isna().any():
					df = self._impute_column(df, col, strategy)
		
		logger.info(f"Imputation completed using strategy: {strategy}")
		return df
	
	def _impute_column(self, df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
		"""Impute missing values for a specific column."""
		series = df[column]
		missing_count = series.isna().sum()
		
		if missing_count == 0:
			return df
		
		if strategy == "auto":
			strategy = self._choose_strategy(series)
		
		if strategy == "mean" and pd.api.types.is_numeric_dtype(series):
			impute_value = series.mean()
		elif strategy == "median" and pd.api.types.is_numeric_dtype(series):
			impute_value = series.median()
		elif strategy == "mode":
			impute_value = series.mode().iloc[0] if not series.mode().empty else series.dropna().iloc[0]
		elif strategy == "forward_fill":
			df[column] = series.fillna(method='ffill')
			return df
		elif strategy == "backward_fill":
			df[column] = series.fillna(method='bfill')
			return df
		else:
			# Default to mode for categorical, mean for numeric
			if pd.api.types.is_numeric_dtype(series):
				impute_value = series.mean()
			else:
				impute_value = series.mode().iloc[0] if not series.mode().empty else series.dropna().iloc[0]
		
		df[column] = series.fillna(impute_value)
		logger.info(f"Imputed {missing_count} missing values in '{column}' using {strategy}")
		return df
	
	def _choose_strategy(self, series: pd.Series) -> str:
		"""Automatically choose the best imputation strategy."""
		if pd.api.types.is_numeric_dtype(series):
			# Check for outliers
			if series.std() > 0:
				z_scores = np.abs((series - series.mean()) / series.std())
				if (z_scores > 3).any():
					return "median"  # Use median for data with outliers
				else:
					return "mean"  # Use mean for normal distribution
			else:
				return "mean"
		else:
			return "mode"  # Use mode for categorical data
