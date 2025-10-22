from typing import Union, Optional
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ArrayType, StringType
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

try:
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = get_logger(__name__)


class LemmatizationAgentPyspark(PySparkBaseAgent):
    """PySpark agent for lemmatizing text tokens in DataFrames."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                tokens_col: str = "filtered_words", output_col: str = "lemmatized_words", 
                **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Lemmatize text tokens in the specified column.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to lemmatize (if None, uses tokens_col parameter)
            tokens_col: Name of the column containing token arrays
            output_col: Name of the output column for lemmatized tokens
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with lemmatized tokens
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        
        # Use column parameter if provided, otherwise use tokens_col
        target_col = column if column is not None else tokens_col
        self.log_processing_start(target_col, tokens_col=tokens_col, output_col=output_col)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        if target_col not in df.columns:
            logger.warning(f"Column '{target_col}' not found in DataFrame")
            return df

        if not NLTK_AVAILABLE:
            logger.error("NLTK not available. Cannot perform lemmatization.")
            return df

        logger.info(f"Lemmatizing tokens in column: {target_col}")

        lemmatizer = WordNetLemmatizer()

        def lemmatize_words(words):
            if not words:
                return []
            return [lemmatizer.lemmatize(w) for w in words]

        lemmatize_udf = F.udf(lemmatize_words, ArrayType(StringType()))
        df = df.withColumn(output_col, lemmatize_udf(F.col(target_col)))

        # Avoid expensive counts; rely on logical logging
        self.log_processing_end(column=target_col)
        
        logger.info(f"Lemmatization complete; output column: {output_col}")
        return df
