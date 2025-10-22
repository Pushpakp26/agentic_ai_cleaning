import pandas as pd

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger


logger = get_logger(__name__)


class DedupAgent(BaseAgent):
	"""Agent for removing duplicate rows from the dataset."""
	
	def __init__(self):
		super().__init__("DedupAgent")
	
	def process(self, df: pd.DataFrame, subset: list = None, keep: str = "first", **kwargs) -> pd.DataFrame:
		"""Remove duplicate rows from the dataframe."""
		initial_rows = len(df)
		
		if subset:
			# Remove duplicates based on specific columns
			df_deduped = df.drop_duplicates(subset=subset, keep=keep)
		else:
			# Remove duplicates based on all columns
			df_deduped = df.drop_duplicates(keep=keep)
		
		removed_count = initial_rows - len(df_deduped)
		
		if removed_count > 0:
			logger.info(f"Removed {removed_count} duplicate rows (kept {keep})")
		else:
			logger.info("No duplicate rows found")
		
		return df_deduped
