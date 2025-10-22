import re
from typing import Union, Optional
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import StringType
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaningPysparkAgent(PySparkBaseAgent):
    """PySpark agent for cleaning text columns in DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                text_col: str = "text", **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Clean text columns by converting to lowercase, removing special characters, and normalizing whitespace.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to clean (if None, uses text_col parameter)
            text_col: Name of the text column to clean
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with cleaned text
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        
        # Use column parameter if provided, otherwise use text_col
        target_col = column if column is not None else text_col
        self.log_processing_start(target_col, text_col=text_col)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        if target_col not in df.columns:
            logger.warning(f"Column '{target_col}' not found in DataFrame")
            return df

        logger.info(f"Cleaning text column: {target_col}")

        def clean_text(text):
            if text is None:
                return ""
            text = text.lower()
            text = re.sub(r"[^a-z\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        clean_udf = F.udf(clean_text, StringType())
        df = df.withColumn(target_col, clean_udf(F.col(target_col)))

        # Avoid expensive counts; rely on logical logging
        self.log_processing_end(column=target_col)
        
        logger.info("Completed text cleaning using regex UDF")
        return df
