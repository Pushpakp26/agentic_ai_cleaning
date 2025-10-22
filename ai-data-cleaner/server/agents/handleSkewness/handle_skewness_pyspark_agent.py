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


class HandleSkewnessAgentPyspark(PySparkBaseAgent):
    """PySpark agent for handling skewed numeric columns."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                threshold: float = 1.0, **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Handle skewed numeric columns by applying log transformation.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to process (if None, processes all numeric columns)
            threshold: Skewness threshold for applying transformation
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with skewness handled
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column, threshold=threshold)
        
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
        
        # Handle skewness for each column
        for col in numeric_cols:
            try:
                skew_val = df.select(F.skewness(F.col(col)).alias("skew")).collect()[0]["skew"]
                if skew_val and abs(skew_val) > threshold:
                    df = df.withColumn(col, F.log1p(F.col(col)))
                    logger.info(f"Applied log1p to column '{col}' (skew={skew_val:.2f})")
                else:
                    logger.info(f"Column '{col}' skewness ({skew_val:.2f}) within threshold")
            except Exception as e:
                logger.warning(f"Could not calculate skewness for column '{col}': {e}")
        
        self.log_processing_end(column=column)
        
        return df
