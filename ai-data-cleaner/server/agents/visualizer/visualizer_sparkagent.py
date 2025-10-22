import base64
import io
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, countDistinct, mean, min as spark_min, max as spark_max

try:
    from ..base_agent import BaseAgent
    from ...config import PROCESSED_DIR
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from config import PROCESSED_DIR
    from utils.logger import get_logger

logger = get_logger(__name__)

class VisualizerAgentSpark(BaseAgent):
    """Visualizer agent for PySpark DataFrames without full conversion to Pandas."""

    def __init__(self):
        super().__init__("VisualizerAgentSpark")
        plt.style.use('seaborn-v0_8')
        self.output_dir = PROCESSED_DIR / "visualizations_spark"
        self.output_dir.mkdir(exist_ok=True)

    def process(self, df: DataFrame, **kwargs) -> Dict[str, Any]:
        visualizations = {}

        # Separate numeric and categorical columns
        numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ("IntegerType", "DoubleType", "FloatType", "LongType")]
        categorical_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType"]

        # Log column types for debugging
        logger.info(f"Processing visualizations: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")

        # Numeric visualizations
        for col_name in numeric_cols:
            try:
                stats = df.select(col_name).agg(
                    spark_min(col_name).alias("min"),
                    spark_max(col_name).alias("max"),
                    mean(col_name).alias("mean")
                ).collect()[0]
                values = df.select(col_name).rdd.flatMap(lambda x: [x[0]]).filter(lambda x: x is not None).take(1000)
                visualizations[f"{col_name}_histogram"] = self._plot_histogram(values, col_name)
                visualizations[f"{col_name}_boxplot"] = self._plot_boxplot(values, col_name)
                logger.debug(f"Generated visualizations for numeric column: {col_name}")
            except Exception as e:
                logger.warning(f"Failed to generate visualization for numeric column '{col_name}': {e}")

        # Categorical visualizations (limit top 5 columns)
        for col_name in categorical_cols[:5]:
            try:
                counts = df.groupBy(col_name).count().orderBy("count", ascending=False).limit(20).collect()
                categories = [str(row[col_name]) if row[col_name] is not None else "<NULL>" for row in counts]
                counts_values = [row["count"] for row in counts]
                visualizations[f"{col_name}_bar"] = self._plot_bar(categories, counts_values, col_name)
                logger.debug(f"Generated visualization for categorical column: {col_name}")
            except Exception as e:
                logger.warning(f"Failed to generate visualization for categorical column '{col_name}': {e}")

        # Correlation heatmap
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self._compute_correlation(df, numeric_cols)
                visualizations["correlation_heatmap"] = self._plot_correlation_heatmap(corr_matrix, numeric_cols)
                logger.debug("Generated correlation heatmap")
            except Exception as e:
                logger.warning(f"Failed to generate correlation heatmap: {e}")

        # Missing values heatmap
        try:
            null_counts = df.select([count(col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
            if any(v > 0 for v in null_counts.values()):
                visualizations["missing_values"] = self._plot_missing_values(null_counts)
                logger.debug("Generated missing values heatmap")
        except Exception as e:
            logger.warning(f"Failed to generate missing values heatmap: {e}")

        logger.info(f"Generated {len(visualizations)} PySpark visualizations")
        return visualizations

    # ----------------------------
    # Plotting methods (base64)
    # ----------------------------
    def _plot_histogram(self, data, column):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return self._figure_to_base64(fig)

    def _plot_boxplot(self, data, column):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.boxplot(data)
        ax.set_title(f'Boxplot of {column}')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        return self._figure_to_base64(fig)

    def _plot_bar(self, categories, counts, column):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(categories, counts)
        ax.set_title(f'Top Categories in {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        return self._figure_to_base64(fig)

    def _compute_correlation(self, df: DataFrame, numeric_cols):
        # Compute correlation between numeric columns using Spark
        corr_matrix = []
        for col1 in numeric_cols:
            row = []
            mean_col1 = df.select(mean(col1)).collect()[0][0]
            std_col1 = df.select(((col(col1) - mean_col1) ** 2).alias("sq")).agg({"sq": "mean"}).collect()[0][0] ** 0.5
            for col2 in numeric_cols:
                mean_col2 = df.select(mean(col2)).collect()[0][0]
                std_col2 = df.select(((col(col2) - mean_col2) ** 2).alias("sq")).agg({"sq": "mean"}).collect()[0][0] ** 0.5
                cov = df.select(((col(col1)-mean_col1)*(col(col2)-mean_col2)).alias("prod")).agg({"prod":"mean"}).collect()[0][0]
                corr = cov / (std_col1 * std_col2) if std_col1*std_col2 !=0 else 0
                row.append(corr)
            corr_matrix.append(row)
        return corr_matrix

    def _plot_correlation_heatmap(self, corr_matrix, numeric_cols):
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True,
                    xticklabels=numeric_cols, yticklabels=numeric_cols, fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap')
        return self._figure_to_base64(fig)

    def _plot_missing_values(self, null_counts):
        fig, ax = plt.subplots(figsize=(12,8))
        cols = list(null_counts.keys())
        counts = list(null_counts.values())
        sns.heatmap([counts], annot=True, fmt="d", cmap="Reds", cbar=True, yticklabels=["Missing"], xticklabels=cols, ax=ax)
        ax.set_title("Missing Values by Column")
        return self._figure_to_base64(fig)

    def _figure_to_base64(self, fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
