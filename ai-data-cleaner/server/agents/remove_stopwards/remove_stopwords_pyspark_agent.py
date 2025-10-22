from typing import Union, Optional
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql import DataFrame
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class RemoveStopwordsAgentPyspark(PySparkBaseAgent):
    """PySpark agent for removing stopwords from text columns."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                text_col: str = "text", output_col: str = "filtered_words", 
                **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Remove stopwords from text columns using Spark ML.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to process (if None, uses text_col parameter)
            text_col: Name of the text column
            output_col: Name of the output column for filtered words
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with stopwords removed
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        
        # Use column parameter if provided, otherwise use text_col
        target_col = column if column is not None else text_col
        self.log_processing_start(target_col, text_col=text_col, output_col=output_col)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        if target_col not in df.columns:
            logger.warning(f"Column '{target_col}' not found in DataFrame")
            return df

        logger.info(f"Removing stopwords from column: {target_col}")

        tokenizer = Tokenizer(inputCol=target_col, outputCol="words")
        tokenized_df = tokenizer.transform(df)

        remover = StopWordsRemover(inputCol="words", outputCol=output_col)
        cleaned_df = remover.transform(tokenized_df)

        # Avoid expensive counts; rely on logical logging
        self.log_processing_end(column=target_col)
        
        logger.info(f"Stopwords removed; output column: {output_col}")
        return cleaned_df
