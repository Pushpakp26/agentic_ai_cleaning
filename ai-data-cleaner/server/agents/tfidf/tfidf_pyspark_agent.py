from typing import Union, Optional
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql import DataFrame, functions as F
import pandas as pd

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class TfidfPsparkAgent(PySparkBaseAgent):
    """PySpark agent for applying TF-IDF vectorization to text columns."""
    
    def process(self, df: Union[DataFrame, pd.DataFrame] = None, column: Optional[str] = None, 
                text_col: str = "text", output_col: str = "tfidf_features", 
                num_features: int = 1000, **kwargs) -> Union[DataFrame, pd.DataFrame]:
        """
        Apply TF-IDF vectorization to text columns.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to process (if None, uses text_col parameter)
            text_col: Name of the text column
            output_col: Name of the output column for TF-IDF features
            num_features: Number of features for hashing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with TF-IDF features
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")
        
        self.validate_input(df)
        
        # Use column parameter if provided, otherwise use text_col
        target_col = column if column is not None else text_col
        self.log_processing_start(target_col, text_col=text_col, output_col=output_col, num_features=num_features)
        
        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)
        
        if target_col not in df.columns:
            logger.warning(f"Column '{target_col}' not found in DataFrame")
            return df

        logger.info(f"Applying TF-IDF on column: {target_col}")

        tokenizer = Tokenizer(inputCol=target_col, outputCol="words")
        words_data = tokenizer.transform(df)

        hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=num_features)
        featurized_data = hashing_tf.transform(words_data)

        idf = IDF(inputCol="raw_features", outputCol=output_col)
        idf_model = idf.fit(featurized_data)
        result = idf_model.transform(featurized_data).drop("words", "raw_features")

        # Avoid expensive counts; rely on logical logging
        self.log_processing_end(column=target_col)
        
        logger.info(f"Generated TF-IDF vectors in column: {output_col}")
        return result
