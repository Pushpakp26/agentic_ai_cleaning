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


class NormalizationAgentPyspark(PySparkBaseAgent):
    """PySpark agent for normalizing numeric columns to [0,1] range."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Normalize numeric columns to [0,1] range using Min-Max scaling.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to normalize (if None, normalizes all numeric columns)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with normalized values
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
                if f.dataType.simpleString() in ["double", "float", "int", "bigint", "long", "short", "byte", "decimal", "tinyint", "smallint"]
            ]
        
        # Avoid expensive counts; rely on logical logging
        
        # Normalize each column
        for col in numeric_cols:
            stats = df.select(
                F.min(F.col(col)).alias("min"), 
                F.max(F.col(col)).alias("max")
            ).collect()[0]
            
            min_val, max_val = stats["min"], stats["max"]
            if min_val is not None and max_val is not None and min_val != max_val:
                df = df.withColumn(col, (F.col(col) - min_val) / (max_val - min_val))
                logger.info(f"Normalized column '{col}' using Min-Max scaling to [0,1] range")
            else:
                logger.warning(f"Column '{col}' has no variation or null values, skipping normalization")
        
        self.log_processing_end(column=column)
        
        return df
