import pandas as pd

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)

class HandleDatetimeAgent(BaseAgent):
    """Agent to extract datetime features from datetime columns."""

    def __init__(self):
        super().__init__("HandleDatetimeAgent")

    def process(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        try:
            df[column] = pd.to_datetime(df[column], errors='coerce')
            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_weekday"] = df[column].dt.weekday
            df[f"{column}_hour"] = df[column].dt.hour
            df = df.drop(columns=[column])
            logger.info(f"Extracted datetime features from '{column}'")
        except Exception as e:
            logger.warning(f"Skipping datetime handling for '{column}': {e}")
        return df
