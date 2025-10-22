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


class ImputersAgent(PySparkBaseAgent):
    """PySpark agent for imputing missing values in DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                strategy: str = "mean", **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Impute missing values in the DataFrame.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to impute (if None, imputes all numeric columns)
            strategy: Imputation strategy ("mean", "median", "mode")
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with imputed values
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        self.log_processing_start(column, strategy=strategy)
        
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
        
        # Impute each column
        for col in numeric_cols:
            if strategy == "mean":
                val = df.select(F.mean(F.col(col))).collect()[0][0]
            elif strategy == "median":
                val = df.approxQuantile(col, [0.5], 0.01)[0]
            elif strategy == "mode":
                val = df.groupBy(col).count().orderBy(F.desc("count")).first()[0]
            else:
                raise ValueError("Strategy must be mean, median, or mode.")
            
            df = df.fillna({col: val})
            logger.info(f"Imputed column '{col}' with {strategy}: {val}")
        
        self.log_processing_end(column=column)
        
        return df
