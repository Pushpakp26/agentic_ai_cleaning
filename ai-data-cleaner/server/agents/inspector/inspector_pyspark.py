from typing import Dict, Any, Union
from pyspark.sql import DataFrame
import pandas as pd

try:
    from ..inspector_interface import InspectorInterface
    from ...utils.spark_sampling import sample_per_column_spark
    from ...utils.logger import get_logger
except ImportError:
    from agents.inspector_interface import InspectorInterface
    from utils.spark_sampling import sample_per_column_spark
    from utils.logger import get_logger

logger = get_logger(__name__)


class PySparkInspectorAgent(InspectorInterface):
    """Base inspector agent for PySpark DataFrames."""

    def __init__(self, spark=None, df=None):
        super().__init__("PySparkInspectorAgent")
        self.spark = spark
        self.df = df  # For backward compatibility
        logger.debug("Initialized PySpark InspectorAgent")

    def process(self, df: Union[DataFrame, pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform inspection using PySpark sampling utilities.
        
        Args:
            df: Input DataFrame (PySpark or Pandas)
            **kwargs: Additional parameters for inspection
            
        Returns:
            Dictionary containing inspection results and suggestions
        """
        if df is None:
            df = self.df
            
        if isinstance(df, DataFrame):
            logger.info("Performing Spark-based inspection and sampling")
            return sample_per_column_spark(df)
        else:
            # Convert Pandas to PySpark for inspection
            logger.info("Converting Pandas DataFrame to PySpark for inspection")
            if self.spark is None:
                raise ValueError("Spark session required for PySpark inspection")
            spark_df = self.spark.createDataFrame(df)
            return sample_per_column_spark(spark_df)
