import re
import pandas as pd
from typing import List

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
    from ..inspector.nlp_utils import is_nlp_column
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger
    from agents.inspector.nlp_utils import is_nlp_column


logger = get_logger(__name__)


class TextPreprocessingAgent(BaseAgent):
	"""Agent for preprocessing text columns."""
	
	def __init__(self):
		super().__init__("TextPreprocessingAgent")
		# Simple stopwords list (in production, use nltk.corpus.stopwords)
		self.stopwords = {
			"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
			"has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
			"to", "was", "will", "with"
		}
	
	def process(self, df: pd.DataFrame, column: str = None, operations: List[str] = None, **kwargs) -> pd.DataFrame:
		"""Preprocess text columns."""
		df = df.copy()
		
		if operations is None:
			operations = ["lowercase", "remove_special_chars", "remove_stopwords", "strip_whitespace"]
		
		if column:
			# Check if this is actually an NLP column, not a short categorical column
			if not is_nlp_column(column, df[column]):
				logger.info(f"Skipping text preprocessing for '{column}' - not an NLP column (likely categorical)")
				return df
			
			if pd.api.types.is_object_dtype(df[column]):
				df = self._preprocess_column(df, column, operations)
		else:
			# Process all text columns
			text_cols = df.select_dtypes(include=['object']).columns
			for col in text_cols:
				# Skip categorical columns
				if not is_nlp_column(col, df[col]):
					logger.info(f"Skipping text preprocessing for '{col}' - not an NLP column (likely categorical)")
					continue
				df = self._preprocess_column(df, col, operations)
		
		logger.info(f"Text preprocessing completed with operations: {operations}")
		return df
	
	def _preprocess_column(self, df: pd.DataFrame, column: str, operations: List[str]) -> pd.DataFrame:
		"""Preprocess a specific text column."""
		series = df[column].astype(str)
		
		for operation in operations:
			if operation == "lowercase":
				series = series.str.lower()
			elif operation == "remove_special_chars":
				series = series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
			elif operation == "remove_stopwords":
				series = series.apply(self._remove_stopwords)
			elif operation == "strip_whitespace":
				series = series.str.strip()
			elif operation == "remove_extra_spaces":
				series = series.str.replace(r'\s+', ' ', regex=True)
			elif operation == "remove_numbers":
				series = series.str.replace(r'\d+', '', regex=True)
		
		df[column] = series
		logger.info(f"Preprocessed text column '{column}' with operations: {operations}")
		return df
	
	def _remove_stopwords(self, text: str) -> str:
		"""Remove stopwords from text."""
		if not isinstance(text, str):
			return text
		
		words = text.split()
		filtered_words = [word for word in words if word.lower() not in self.stopwords]
		return ' '.join(filtered_words)
