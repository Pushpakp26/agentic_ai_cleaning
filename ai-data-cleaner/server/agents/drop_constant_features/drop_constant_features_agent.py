import pandas as pd

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)

class DropConstantFeaturesAgent(BaseAgent):
    """Agent to drop constant columns with only one unique value."""

    def __init__(self):
        super().__init__("DropConstantFeaturesAgent")

    def process(self, df: pd.DataFrame, column: str = None, **kwargs) -> pd.DataFrame:
        const_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            logger.info(f"Dropped constant columns: {const_cols}")
        return df
