from typing import Union, Optional, List
from pyspark.sql import DataFrame, functions as F
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class DropConstantFeaturesAgentPyspark(PySparkBaseAgent):
    """PySpark agent for dropping constant features from DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Drop constant features (columns with only one unique value).
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to check (if None, checks all columns)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with constant columns removed
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        # Determine which columns to check
        if column:
            cols_to_check = [column]
        else:
            cols_to_check = df.columns
        
        # Avoid expensive counts; rely on logical logging
        constant_cols = []
        
        # Check each column for constant values
        for col in cols_to_check:
            if col in df.columns:
                try:
                    unique_count = df.select(F.countDistinct(F.col(col))).collect()[0][0]
                    if unique_count == 1:
                        constant_cols.append(col)
                        logger.info(f"Column '{col}' is constant (1 unique value)")
                except Exception as e:
                    logger.warning(f"Could not check column '{col}': {e}")
        
        # Drop constant columns
        if constant_cols:
            df = df.drop(*constant_cols)
            logger.info(f"Dropped constant columns: {constant_cols}")
        else:
            logger.info("No constant columns found")
        
        self.log_processing_end(column=column)
        
        return df
