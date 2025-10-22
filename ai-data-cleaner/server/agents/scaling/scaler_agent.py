import pandas as pd
import numpy as np

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger


logger = get_logger(__name__)


class ScalerAgent(BaseAgent):
    """Agent for scaling numeric columns using standardization or other methods."""
    
    def __init__(self):
        super().__init__("ScalerAgent")
    
    def process(self, df: pd.DataFrame, column: str = None, method: str = "standard", **kwargs) -> pd.DataFrame:
        """Scale numeric columns using various methods."""
        df = df.copy()
        
        if column:
            if pd.api.types.is_numeric_dtype(df[column]):
                df = self._scale_column(df, column, method, **kwargs)
        else:
            # Scale all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df = self._scale_column(df, col, method, **kwargs)
        
        logger.info(f"Scaling completed using method: {method}")
        return df
    
    def _scale_column(self, df: pd.DataFrame, column: str, method: str, **kwargs) -> pd.DataFrame:
        """Scale a specific numeric column."""
        series = df[column]
        
        if method == "standard":
            # Z-score standardization
            mean_val, std_val = series.mean(), series.std()
            if std_val != 0:
                df[column] = (series - mean_val) / std_val
            else:
                df[column] = 0  # Handle case where all values are the same
        
        elif method == "robust":
            # Use median and IQR for robustness to outliers
            median_val = series.median()
            q75, q25 = series.quantile(0.75), series.quantile(0.25)
            iqr = q75 - q25
            if iqr != 0:
                df[column] = (series - median_val) / iqr
            else:
                df[column] = 0
        
        elif method == "remove_outliers":
            # Remove outliers using IQR method
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Strategy to handle outliers: 'clip' (default), 'nan', or 'median'
            strategy = kwargs.get("outlier_strategy", "clip")
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            
            if outlier_count > 0:
                if strategy == "clip":
                    df[column] = series.clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Clipped {outlier_count} outliers to IQR bounds in column '{column}'")
                elif strategy == "median":
                    median_val = series.median()
                    df.loc[outlier_mask, column] = median_val
                    logger.info(f"Replaced {outlier_count} outliers with median in column '{column}'")
                else:
                    # Set outliers to NaN (only if explicitly requested)
                    df.loc[outlier_mask, column] = np.nan
                    logger.info(f"Set {outlier_count} outliers to NaN in column '{column}'")
        
        logger.info(f"Scaled column '{column}' using {method} method")
        return df
