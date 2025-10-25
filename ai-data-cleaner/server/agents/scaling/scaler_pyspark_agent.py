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


class ScalingAgentPyspark(PySparkBaseAgent):
    """PySpark agent for scaling/normalizing numeric columns in DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                method: str = "standardize", **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Scale/normalize numeric columns in the DataFrame.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to scale (if None, scales all numeric columns)
            method: Scaling method - "standardize"/"standard" (Z-score), 
                   "normalize"/"minmax" (0-1 range), or "robust" (median/IQR)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with scaled values
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column, method=method)
        
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
        
        # Scale each column
        for col in numeric_cols:
            # Support both naming conventions: 'standard'/'standardize', 'minmax'/'normalize'
            if method in ["standardize", "standard"]:
                stats = df.select(
                    F.mean(F.col(col)).alias("mean"), 
                    F.stddev(F.col(col)).alias("std")
                ).collect()[0]
                if stats["std"] and stats["std"] != 0:
                    df = df.withColumn(col, (F.col(col) - stats["mean"]) / stats["std"])
                    logger.info(f"Standardized column '{col}' using Z-score scaling")
            elif method in ["normalize", "minmax"]:
                stats = df.select(
                    F.min(F.col(col)).alias("min"), 
                    F.max(F.col(col)).alias("max")
                ).collect()[0]
                if stats["max"] != stats["min"]:
                    df = df.withColumn(col, (F.col(col) - stats["min"]) / (stats["max"] - stats["min"]))
                    logger.info(f"Normalized column '{col}' to [0,1] range")
            elif method == "robust":
                stats = df.select(
                    F.expr(f"percentile_approx({col}, 0.5)").alias("median"),
                    F.expr(f"percentile_approx({col}, 0.75) - percentile_approx({col}, 0.25)").alias("iqr")
                ).collect()[0]
                if stats["iqr"] and stats["iqr"] != 0:
                    df = df.withColumn(col, (F.col(col) - stats["median"]) / stats["iqr"])
                    logger.info(f"Applied robust scaling to column '{col}'")
            else:
                raise ValueError(f"Method must be 'standardize'/'standard', 'normalize'/'minmax', or 'robust'. Got: '{method}'")
        
        self.log_processing_end(column=column)
        
        return df
