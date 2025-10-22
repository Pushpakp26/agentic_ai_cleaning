from __future__ import annotations
from typing import Dict, Any
from pyspark.sql import DataFrame, functions as F

try:
    from ..config import SAMPLE_ROWS_PER_COLUMN, MAX_UNIQUE_PREVIEW
except ImportError:
    from config import SAMPLE_ROWS_PER_COLUMN, MAX_UNIQUE_PREVIEW


def profile_dataframe_spark(df: DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return per-column profile stats using PySpark operations."""
    profile: Dict[str, Dict[str, Any]] = {}
    total_rows = df.count()

    for col, dtype in df.dtypes:
        info: Dict[str, Any] = {"dtype": dtype}

        # Count nulls and non-nulls
        nulls = df.filter(F.col(col).isNull()).count()
        non_null = total_rows - nulls
        info.update({"non_null": non_null, "nulls": nulls})

        # Numeric columns
        if dtype in ("int", "bigint", "double", "float", "decimal"):
            stats = (
                df.select(
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                    F.mean(col).alias("mean"),
                    F.stddev(col).alias("std"),
                )
                .collect()[0]
            )
            info.update({
                "min": _try_float(stats["min"]),
                "max": _try_float(stats["max"]),
                "mean": _try_float(stats["mean"]),
                "std": _try_float(stats["std"]),
            })
        else:
            # For string or categorical columns
            distinct_count = df.select(col).distinct().count()
            sample_values = (
                df.select(col)
                .where(F.col(col).isNotNull())
                .distinct()
                .limit(MAX_UNIQUE_PREVIEW)
                .toPandas()[col]
                .astype(str)
                .tolist()
            )
            info.update({
                "unique_count": distinct_count,
                "sample_unique": sample_values,
            })
        profile[col] = info
    return profile


def sample_per_column_spark(df: DataFrame, rows: int = 5) -> Dict[str, Any]:
    """Collect up to N non-null samples per column alongside profiles."""
    samples: Dict[str, Any] = {}

    for col, _ in df.dtypes:
        # Sample non-null rows for this column
        sample_list = (
            df.select(col)
            .where(F.col(col).isNotNull())
            .limit(rows)
            .toPandas()[col]
            .tolist()
        )
        samples[col] = sample_list

    return {
        "profile": profile_dataframe_spark(df),
        "samples": samples
    }


def _try_float(value):
    try:
        return float(value)
    except Exception:
        return None
