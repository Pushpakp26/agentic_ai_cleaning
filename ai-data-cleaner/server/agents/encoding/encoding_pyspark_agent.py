from typing import Union, Optional, List
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

try:
    from ..pyspark_base_agent import PySparkBaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.pyspark_base_agent import PySparkBaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)


class EncodingAgentPyspark(PySparkBaseAgent):
    """PySpark agent for encoding categorical columns using one-hot or label encoding."""

    def process(
        self,
        df: Union[DataFrame, pd.DataFrame] = None,
        column: Optional[str] = None,
        method: str = "onehot",
        strategy: Optional[str] = None,
        return_metadata: bool = False,
        **kwargs,
    ) -> Union[DataFrame, pd.DataFrame]:
        """
        Encode categorical columns using one-hot or label encoding.

        Args:
            df: Input DataFrame (PySpark or Pandas)
            column: Specific column to encode (if None, encodes all string/category columns)
            method: 'onehot' | 'label' | 'categorical_encode' (defaults to 'onehot')
            strategy: When method == 'categorical_encode', choose 'onehot' or 'label' (defaults to 'onehot')
            return_metadata: If True, return (df, metadata) tuple for orchestrator tracking
        """
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No DataFrame provided")

        # Normalize method
        enc_method = method
        if method == "categorical_encode":
            enc_method = strategy or "onehot"
        if enc_method not in ("onehot", "label"):
            raise ValueError("method/strategy must be 'onehot' or 'label'")

        self.validate_input(df)
        self.log_processing_start(column, method=enc_method)

        # Convert to PySpark if needed
        if isinstance(df, pd.DataFrame):
            if self.spark is None:
                raise ValueError("Spark session required for PySpark processing")
            df = self.spark.createDataFrame(df)

        # Determine target columns
        target_cols: List[str] = [column] if column else [
            f.name for f in df.schema.fields if f.dataType.simpleString() == "string"
        ]
        if not target_cols:
            logger.info("No categorical (string) columns found for encoding")
            return df

        # Track encoding metadata for orchestrator
        encoding_metadata = {
            'method': enc_method,
            'encoded_columns': {}
        }

        if enc_method == "label":
            for col in target_cols:
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}__idx", handleInvalid="keep")
                model = indexer.fit(df)
                df = model.transform(df)
                df = df.drop(col).withColumnRenamed(f"{col}__idx", col)
                logger.info(f"Applied label encoding on column '{col}'")
                
                # Track metadata: column replaced with numeric version in-place
                encoding_metadata['encoded_columns'][col] = {
                    'new_columns': [col],  # Same column name, now numeric
                    'removed': False  # Not removed, transformed in-place
                }
        else:
            index_output_cols = [f"{c}__idx" for c in target_cols]
            ohe_output_cols = [f"{c}_ohe" for c in target_cols]

            stages = [
                StringIndexer(inputCol=inp, outputCol=out, handleInvalid="keep")
                for inp, out in zip(target_cols, index_output_cols)
            ]
            stages.append(
                OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols, handleInvalid="keep")
            )
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(df)
            df = model.transform(df)
            # Drop original categorical columns and intermediate index columns
            for orig, idx_col in zip(target_cols, index_output_cols):
                df = df.drop(orig, idx_col)
            logger.info(f"Applied one-hot encoding on columns: {target_cols}")
            
            # Track metadata: original columns replaced with OHE columns
            for orig, ohe_col in zip(target_cols, ohe_output_cols):
                encoding_metadata['encoded_columns'][orig] = {
                    'new_columns': [ohe_col],  # Vector column
                    'removed': True
                }

        self.log_processing_end(column=column)
        
        if return_metadata:
            return df, encoding_metadata
        return df
