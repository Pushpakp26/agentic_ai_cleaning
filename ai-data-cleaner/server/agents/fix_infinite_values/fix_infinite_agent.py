import pandas as pd
import numpy as np

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)

class FixInfiniteAgent(BaseAgent):
    """Agent to replace infinite values with NaN."""

    def __init__(self):
        super().__init__("FixInfiniteAgent")

    def process(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        inf_count = np.isinf(df[column]).sum()
        if inf_count > 0:
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)
            logger.info(f"Replaced {inf_count} infinite values in '{column}' with NaN")
        return df
