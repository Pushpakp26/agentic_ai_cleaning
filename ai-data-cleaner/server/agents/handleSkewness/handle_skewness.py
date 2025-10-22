import pandas as pd
import numpy as np

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)

class HandleSkewnessAgent(BaseAgent):
    """Agent to correct skewed numeric columns using log transform."""

    def __init__(self):
        super().__init__("HandleSkewnessAgent")

    def process(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        if not np.issubdtype(df[column].dtype, np.number):
            return df

        skewness = df[column].skew()
        if abs(skewness) > 1:
            min_val = df[column].min()
            if min_val <= 0:
                df[column] = np.log1p(df[column] - min_val + 1)
            else:
                df[column] = np.log1p(df[column])
            logger.info(f"Handled skewness in column '{column}' (skew={skewness:.2f})")
        return df
