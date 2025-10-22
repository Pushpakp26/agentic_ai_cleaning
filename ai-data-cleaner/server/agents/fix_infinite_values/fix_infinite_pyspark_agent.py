from typing import Union, Optional
from pyspark.sql import DataFrame, functions as F
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class FixInfiniteValuesAgentPyspark(PySparkBaseAgent):
    """PySpark agent for fixing infinite and NaN values in numeric columns."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Fix infinite and NaN values in numeric columns.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to fix (if None, fixes all numeric columns)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with infinite values fixed
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
        
        # Get numeric columns
        if column:
            numeric_cols = [column]
        else:
            numeric_cols = [
                f.name for f in df.schema.fields
                if f.dataType.simpleString() in ["double", "float", "int", "bigint"]
            ]
        
        # Avoid expensive counts; rely on logical logging
        
        # Fix infinite values for each column
        for col in numeric_cols:
            df = df.withColumn(
                col,
                F.when(
                    F.isnan(F.col(col)) | F.isnull(F.col(col)) |
                    (F.col(col) == float("inf")) | (F.col(col) == float("-inf")),
                    None
                ).otherwise(F.col(col))
            )
            logger.info(f"Fixed infinite/NaN values in column: {col}")
        
        self.log_processing_end(column=column)
        
        logger.info("Replaced NaN and Infinite values using F.when()")
        return df
