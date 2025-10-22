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


class HandleDatetimeAgentPyspark(PySparkBaseAgent):
    """PySpark agent for handling datetime columns."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                datetime_cols: Optional[List[str]] = None, **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Handle datetime columns by converting to timestamp and extracting components.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to process (if None, processes all datetime columns)
            datetime_cols: List of datetime column names to process
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with datetime handling applied
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column, datetime_cols=datetime_cols)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        # Determine which columns to process
        if column:
            target_cols = [column]
        elif datetime_cols:
            target_cols = datetime_cols
        else:
            # Auto-detect datetime columns
            target_cols = [
                col for col in df.columns 
                if any(k in col.lower() for k in ["date", "time", "timestamp"])
            ]
        
        # Avoid expensive counts; rely on logical logging
        
        # Process each datetime column
        for col in target_cols:
            if col in df.columns:
                try:
                    df = df.withColumn(col, F.to_timestamp(F.col(col), "yyyy-MM-dd HH:mm:ss"))
                    df = (
                        df
                        .withColumn(f"{col}_year", F.year(F.col(col)))
                        .withColumn(f"{col}_month", F.month(F.col(col)))
                        .withColumn(f"{col}_day", F.dayofmonth(F.col(col)))
                    )
                    logger.info(f"Extracted datetime parts for column: {col}")
                except Exception as e:
                    logger.warning(f"Could not process datetime column '{col}': {e}")
            else:
                logger.warning(f"Column '{col}' not found in DataFrame")
        
        self.log_processing_end(column=column)
        
        return df
