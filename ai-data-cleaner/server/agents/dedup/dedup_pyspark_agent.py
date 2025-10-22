from typing import Union, Optional, List
from pyspark.sql import DataFrame
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class DedupAgentSpark(PySparkBaseAgent):
    """PySpark agent for removing duplicate rows from DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                subset: Optional[List[str]] = None, **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to check for duplicates (if None, checks all columns)
            subset: List of column names to consider for duplicate detection
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with duplicates removed
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column, subset=subset)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        # Avoid expensive counts; rely on logical logging
        
        # Determine subset for duplicate detection
        if subset is None and column is not None:
            subset = [column]
        
        # Remove duplicates
        if subset:
            df = df.dropDuplicates(subset)
            logger.info(f"Removed duplicates based on columns: {subset}")
        else:
            df = df.dropDuplicates()
            logger.info("Removed duplicates based on all columns")
        
        # Skip output row counts; log action only
        self.log_processing_end(column=column)
        
        return df
