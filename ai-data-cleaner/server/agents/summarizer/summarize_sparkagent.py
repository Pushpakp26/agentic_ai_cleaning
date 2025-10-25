from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

try:
    from ..base_agent import BaseAgent
    from ...config import PROCESSED_DIR
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from config import PROCESSED_DIR
    from utils.logger import get_logger

logger = get_logger(__name__)


class SummarizerAgentSpark(BaseAgent):
    """Summarizer agent for PySpark DataFrames."""

    def __init__(self):
        super().__init__("SummarizerAgentSpark")
        self.output_dir = PROCESSED_DIR
        self.output_dir.mkdir(exist_ok=True)

    def process(
        self,
        df: DataFrame,
        inspection_results: Dict = None,
        suggestions: Dict = None,
        snapshots: list = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate summary report for PySpark DataFrame."""

        # Defensive normalization
        suggestions = self._normalize_suggestions(suggestions)
        snapshots = snapshots or []
        applied_operations = kwargs.get("applied_operations")

        report_data = {
            "metadata": self._generate_metadata(df),
            "data_overview": self._generate_data_overview(df),
            "column_analysis": self._generate_column_analysis(df),
            "preprocessing_summary": self._generate_preprocessing_summary(suggestions, snapshots, applied_operations),
            "quality_assessment": self._generate_quality_assessment(df),
            "recommendations": self._generate_recommendations(df, inspection_results),
        }

        html_content = self._generate_html_report(report_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_spark_{timestamp}.html"
        report_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Generated Spark summary report: {report_path}")

        return {
            "report_path": str(report_path),
            "report_data": report_data,
            "html_content": html_content,
        }

    # ----------------------------------------------------------------------
    # Helpers: normalization & small utilities
    # ----------------------------------------------------------------------
    def _normalize_suggestions(self, suggestions):
        """
        Ensure `suggestions` is a dict mapping column -> suggestion-dict.
        Accepts:
         - None -> {}
         - dict -> returned as-is
         - list -> tries to convert list items into mapping
             * If each item is a dict and has 'column' or 'col' key, use that
             * If each item is a (col, suggestion) tuple, convert
             * Otherwise return empty dict
        """
        if not suggestions:
            return {}

        # Already a dict
        if isinstance(suggestions, dict):
            return suggestions

        normalized: Dict[str, Dict] = {}
        if isinstance(suggestions, list):
            for item in suggestions:
                # case: list of dicts with explicit column key
                if isinstance(item, dict):
                    col = None
                    if "column" in item:
                        col = item["column"]
                        # suggestion payload might be nested under 'suggestion'
                        payload = item.get("suggestion") or item.get("operation") or item
                        # Make sure payload is a dict
                        if not isinstance(payload, dict):
                            payload = {"suggestion": payload}
                        normalized[col] = payload
                    elif "col" in item:
                        col = item["col"]
                        payload = item.get("suggestion") or item
                        if not isinstance(payload, dict):
                            payload = {"suggestion": payload}
                        normalized[col] = payload
                    elif len(item) == 1 and isinstance(next(iter(item.keys())), str):
                        # maybe {'Age': {'suggestion': 'standardize'}}
                        key = next(iter(item.keys()))
                        if isinstance(item[key], dict):
                            normalized[key] = item[key]
                        else:
                            normalized[key] = {"suggestion": item[key]}
                    else:
                        # fallback: try to find first string key
                        for k, v in item.items():
                            if isinstance(k, str):
                                if isinstance(v, dict):
                                    normalized[k] = v
                                else:
                                    normalized[k] = {"suggestion": v}
                                break
                # case: tuple (col, payload)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    col = item[0]
                    payload = item[1]
                    if isinstance(payload, dict):
                        normalized[col] = payload
                    else:
                        normalized[col] = {"suggestion": payload}
                else:
                    # unknown shape: skip
                    continue
        else:
            # unknown type (e.g., string) -> return empty
            return {}

        return normalized

    # ----------------------------------------------------------------------
    # Report generation helpers
    # ----------------------------------------------------------------------

    def _generate_metadata(self, df: DataFrame) -> Dict[str, Any]:
        """Generate dataset metadata for PySpark DataFrame."""
        # For Spark, computing precise memory usage for the DataFrame is non-trivial.
        # Provide a safe placeholder to avoid KeyErrors in HTML generation.
        try:
            row_count = df.count()
        except Exception:
            row_count = None

        try:
            col_count = len(df.columns)
        except Exception:
            col_count = None

        dtypes = {}
        try:
            dtypes = {f.name: f.dataType.simpleString() for f in df.schema.fields}
        except Exception:
            dtypes = {}

        return {
            "timestamp": datetime.now().isoformat(),
            "shape": (row_count or 0, col_count or 0),
            "columns": df.columns if hasattr(df, "columns") else [],
            "dtypes": dtypes,
            # Not calculating actual memory - set to 'N/A' for Spark
            "memory_usage": "N/A",
        }

    def _generate_data_overview(self, df: DataFrame) -> Dict[str, Any]:
        """Generate high-level data overview."""
        total_rows = df.count()
        total_columns = len(df.columns)

        missing_counts_row = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()
        missing_counts = {}
        if missing_counts_row:
            missing_counts = missing_counts_row[0].asDict()
        total_missing = sum(missing_counts.values()) if missing_counts else 0
        missing_percentage = (total_missing / (total_rows * total_columns) * 100) if total_rows > 0 else 0

        duplicate_rows = total_rows - df.dropDuplicates().count()

        numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["int", "bigint", "double", "float", "decimal", "long", "short", "byte", "tinyint", "smallint"]]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": total_missing,
            "missing_percentage": f"{missing_percentage:.2f}%",
            "duplicate_rows": duplicate_rows,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
        }

    def _generate_column_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Generate column-level analysis using PySpark ops."""
        analysis = {}
        total_rows = df.count()

        for field in df.schema.fields:
            col = field.name
            dtype = field.dataType.simpleString()

            stats_row = df.select(
                F.count(F.when(F.col(col).isNotNull(), col)).alias("non_null"),
                F.count(F.when(F.col(col).isNull(), col)).alias("null_count"),
            ).collect()
            stats = stats_row[0].asDict() if stats_row else {"non_null": 0, "null_count": 0}

            non_null = stats.get("non_null", 0)
            null_count = stats.get("null_count", 0)
            null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

            unique_count = df.select(col).distinct().count()
            unique_percentage = (unique_count / total_rows * 100) if total_rows > 0 else 0

            col_info = {
                "dtype": dtype,
                "non_null": non_null,
                "null_count": null_count,
                "null_percentage": f"{null_percentage:.2f}%",
                "unique_count": unique_count,
                "unique_percentage": f"{unique_percentage:.2f}%",
            }

            if dtype in ["int", "bigint", "double", "float", "decimal"]:
                # Use safe aggregation
                try:
                    agg_row = df.select(
                        F.mean(col).alias("mean"),
                        F.expr(f"percentile_approx({col}, 0.5)").alias("median"),
                        F.stddev(col).alias("std"),
                        F.min(col).alias("min"),
                        F.max(col).alias("max"),
                    ).collect()
                    agg = agg_row[0].asDict() if agg_row else {}
                except Exception:
                    agg = {}
                col_info.update(agg)
            else:
                try:
                    top_values = df.groupBy(col).count().orderBy(F.desc("count")).limit(5)
                    col_info["top_values"] = {r[col]: r["count"] for r in top_values.collect() if r[col] is not None}
                except Exception:
                    col_info["top_values"] = {}

            analysis[col] = col_info

        return analysis

    def _generate_preprocessing_summary(self, suggestions: Dict = None, snapshots: list = None, applied_operations: list = None) -> Dict[str, Any]:
        """Generate summary of preprocessing steps."""
        # suggestions is already normalized by process()
        if not suggestions and not applied_operations:
            return {"message": "No preprocessing details available", "total_suggestions": 0, "applied_operations": [], "skipped_columns": [], "total_snapshots": 0, "snapshot_steps": []}

        summary = {
            "total_suggestions": len(suggestions),
            "applied_operations": [],
            "skipped_columns": [],
        }

        if applied_operations:
            summary["applied_operations"] = list(applied_operations)
        else:
            for col, suggestion in suggestions.items():
                # suggestion might be None or not a dict; coerce
                if suggestion is None:
                    suggestion = {}
                if isinstance(suggestion, dict):
                    s_value = suggestion.get("suggestion") or suggestion.get("operation") or suggestion.get("action") or suggestion
                    if isinstance(s_value, dict):
                        op_name = s_value.get("suggestion") or s_value.get("operation") or str(s_value)
                    else:
                        op_name = str(s_value)
                    if op_name == "skip" or suggestion.get("suggestion") == "skip":
                        summary["skipped_columns"].append(col)
                    else:
                        summary["applied_operations"].append(
                            {
                                "column": col,
                                "operation": op_name,
                                "reason": suggestion.get("reason", ""),
                            }
                        )
                else:
                    # non-dict suggestion: treat as operation name
                    if str(suggestion).lower() == "skip":
                        summary["skipped_columns"].append(col)
                    else:
                        summary["applied_operations"].append(
                            {"column": col, "operation": str(suggestion), "reason": ""}
                        )

        if snapshots:
            # snapshots expected to be list of dicts with 'step' key; be defensive
            summary["total_snapshots"] = len(snapshots)
            steps = []
            for s in snapshots:
                if isinstance(s, dict) and "step" in s:
                    steps.append(s["step"])
                elif isinstance(s, str):
                    steps.append(s)
                else:
                    try:
                        steps.append(str(s.get("step"))) if hasattr(s, "get") else steps.append(str(s))
                    except Exception:
                        steps.append(str(s))
            summary["snapshot_steps"] = steps
        else:
            summary["total_snapshots"] = 0
            summary["snapshot_steps"] = []

        return summary

    def _generate_quality_assessment(self, df: DataFrame) -> Dict[str, Any]:
        """Generate data quality assessment for PySpark DataFrame."""
        total_rows = df.count()
        total_cols = len(df.columns)

        missing_counts_row = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()
        missing_counts = missing_counts_row[0].asDict() if missing_counts_row else {}
        missing_pct = (sum(missing_counts.values()) / (total_rows * total_cols) * 100) if total_rows > 0 and missing_counts else 0

        dup_count = total_rows - df.dropDuplicates().count()
        dup_pct = (dup_count / total_rows * 100) if total_rows > 0 else 0

        quality_score = 100
        issues = []

        if missing_pct > 10:
            quality_score -= 20
            issues.append(f"High missing values: {missing_pct:.1f}%")
        elif missing_pct > 5:
            quality_score -= 10
            issues.append(f"Moderate missing values: {missing_pct:.1f}%")

        if dup_pct > 5:
            quality_score -= 15
            issues.append(f"High duplicate rate: {dup_pct:.1f}%")
        elif dup_pct > 1:
            quality_score -= 5
            issues.append(f"Moderate duplicate rate: {dup_pct:.1f}%")

        constant_cols = []
        try:
            constant_cols = [c for c in df.columns if df.select(c).distinct().count() <= 1]
        except Exception:
            constant_cols = []

        if constant_cols:
            quality_score -= 10
            issues.append(f"Constant columns found: {constant_cols}")

        assessment = (
            "Excellent" if quality_score >= 90
            else "Good" if quality_score >= 70
            else "Fair" if quality_score >= 50
            else "Poor"
        )

        return {
            "quality_score": max(quality_score, 0),
            "issues": issues,
            "assessment": assessment,
        }

    def _generate_recommendations(self, df: DataFrame, inspection_results: Dict = None) -> list:
        """Generate data recommendations for PySpark DataFrame."""
        recommendations = []

        numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["int", "bigint", "double", "float", "decimal", "long", "short", "byte", "tinyint", "smallint"]]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        total = df.count() if df is not None else 0

        for c in categorical_cols:
            try:
                uniq = df.select(c).distinct().count()
            except Exception:
                uniq = 0
            if total > 0 and uniq > total * 0.5:
                recommendations.append(f"Consider feature engineering for high-cardinality column: {c}")

        for c in numeric_cols:
            try:
                stats_row = df.select(
                    F.mean(c).alias("mean"),
                    F.stddev(c).alias("std"),
                    F.min(c).alias("min"),
                    F.max(c).alias("max"),
                    F.skewness(c).alias("skew"),
                ).collect()
                stats = stats_row[0].asDict() if stats_row else {}
            except Exception:
                stats = {}
            skew_val = stats.get("skew", 0) or 0
            try:
                if abs(float(skew_val)) > 2:
                    recommendations.append(f"Column '{c}' is highly skewed (skewness: {float(skew_val):.2f}). Consider transformation.")
            except Exception:
                continue

        if not recommendations:
            recommendations.append("Dataset appears to be in good condition for analysis.")

        return recommendations

    # ----------------------------------------------------------------------
    # HTML generation (same style as pandas version)
    # ----------------------------------------------------------------------
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        # Make sure all expected keys exist to avoid KeyErrors
        metadata = report_data.get("metadata", {})
        data_overview = report_data.get("data_overview", {})
        column_analysis = report_data.get("column_analysis", {})
        preprocessing_summary = report_data.get("preprocessing_summary", {})
        quality_assessment = report_data.get("quality_assessment", {})
        recommendations = report_data.get("recommendations", [])

        timestamp_str = metadata.get("timestamp", "")[:10]
        shape = metadata.get("shape", (0, 0))
        memory_usage = metadata.get("memory_usage", "N/A")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Cleaning Report - {timestamp_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .quality-score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
        .issue {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .recommendation {{ background-color: #d1ecf1; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .operation {{ background-color: #e8f5e8; padding: 8px; margin: 5px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Cleaning & Preprocessing Report</h1>
        
        <div class="metric">
            <h3>📅 Generated on: {metadata.get('timestamp', '')}</h3>
            <h3>📈 Dataset Shape: {shape[0]:,} rows × {shape[1]} columns</h3>
            <h3>💾 Memory Usage: {memory_usage}</h3>
        </div>
        
        <h2>📋 Data Overview</h2>
        <div class="metric">
            <p><strong>Total Rows:</strong> {data_overview.get('total_rows', 0):,}</p>
            <p><strong>Total Columns:</strong> {data_overview.get('total_columns', 0)}</p>
            <p><strong>Missing Values:</strong> {data_overview.get('missing_values', 0):,} ({data_overview.get('missing_percentage', '0.00%')})</p>
            <p><strong>Duplicate Rows:</strong> {data_overview.get('duplicate_rows', 0):,}</p>
            <p><strong>Numeric Columns:</strong> {data_overview.get('numeric_columns', 0)}</p>
            <p><strong>Categorical Columns:</strong> {data_overview.get('categorical_columns', 0)}</p>
        </div>
        
        <h2>🔍 Column Analysis</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Non-Null</th>
                <th>Missing %</th>
                <th>Unique Count</th>
                <th>Summary</th>
            </tr>
"""
        # Column rows
        for col, info in column_analysis.items():
            summary = ""
            try:
                if 'mean' in info and info.get('mean') is not None:
                    summary = f"Mean: {info['mean']:.2f}, Range: {info.get('min', 0):.2f} - {info.get('max', 0):.2f}"
                elif 'top_values' in info and info['top_values']:
                    top_val = list(info['top_values'].keys())[0]
                    summary = f"Top value: {top_val} ({info['top_values'][top_val]} occurrences)"
            except Exception:
                summary = ""

            html += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{info.get('dtype', '')}</td>
                <td>{info.get('non_null', 0):,}</td>
                <td>{info.get('null_percentage', '0.00%')}</td>
                <td>{info.get('unique_count', 0):,}</td>
                <td>{summary}</td>
            </tr>
"""

        html += f"""
        </table>
        
        <h2>⚙️ Preprocessing Summary</h2>
        <div class="metric">
            <p><strong>Total Suggestions:</strong> {preprocessing_summary.get('total_suggestions', 0)}</p>
            <p><strong>Applied Operations:</strong> {len(preprocessing_summary.get('applied_operations', []))}</p>
            <p><strong>Skipped Columns:</strong> {len(preprocessing_summary.get('skipped_columns', []))}</p>
        </div>
        
        <h3>Applied Operations:</h3>
"""
        # Applied operations
        for op in preprocessing_summary.get('applied_operations', []):
            coln = op.get('column', '')
            opn = op.get('operation', '')
            reason = op.get('reason', '')
            html += f"""
        <div class="operation">
            <strong>{coln}:</strong> {opn} - {reason}
        </div>
"""

        html += f"""
        <h2>🎯 Data Quality Assessment</h2>
        <div class="metric">
            <div class="quality-score">Quality Score: {quality_assessment.get('quality_score', 0)}/100</div>
            <p><strong>Assessment:</strong> {quality_assessment.get('assessment', '')}</p>
        </div>
"""
        if quality_assessment.get('issues'):
            html += "<h3>⚠️ Issues Found:</h3>"
            for issue in quality_assessment.get('issues', []):
                html += f"<div class='issue'>{issue}</div>"

        html += """
        <h2>💡 Recommendations</h2>
"""
        for rec in recommendations:
            html += f"<div class='recommendation'>{rec}</div>"

        html += """
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #666;">
            Report generated by AI Data Cleaner | 
            <a href="#" onclick="window.print()">Print Report</a>
        </p>
    </div>
</body>
</html>
"""
        return html







#below code is of cursor ,nor working but ok code
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, Any

# from pyspark.sql import DataFrame
# from pyspark.sql import functions as F

# try:
#     from ..base_agent import BaseAgent
#     from ...config import PROCESSED_DIR
#     from ...utils.logger import get_logger
# except ImportError:
#     from agents.base_agent import BaseAgent
#     from config import PROCESSED_DIR
#     from utils.logger import get_logger

# logger = get_logger(__name__)


# class SummarizerAgentSpark(BaseAgent):
#     """Summarizer agent for PySpark DataFrames."""

#     def __init__(self):
#         super().__init__("SummarizerAgentSpark")
#         self.output_dir = PROCESSED_DIR
#         self.output_dir.mkdir(exist_ok=True)

#     def process(
#         self,
#         df: DataFrame,
#         inspection_results: Dict = None,
#         suggestions: Dict = None,
#         snapshots: list = None,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """Generate summary report for PySpark DataFrame."""

#         report_data = {
#             "metadata": self._generate_metadata(df),
#             "data_overview": self._generate_data_overview(df),
#             "column_analysis": self._generate_column_analysis(df),
#             "preprocessing_summary": self._generate_preprocessing_summary(suggestions, snapshots),
#             "quality_assessment": self._generate_quality_assessment(df),
#             "recommendations": self._generate_recommendations(df, inspection_results),
#         }

#         html_content = self._generate_html_report(report_data)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         report_path = self.output_dir / f"report_spark_{timestamp}.html"
#         report_path.write_text(html_content, encoding="utf-8")

#         logger.info(f"Generated Spark summary report: {report_path}")

#         return {
#             "report_path": str(report_path),
#             "report_data": report_data,
#             "html_content": html_content,
#         }

#     # ----------------------------------------------------------------------
#     # Report generation helpers
#     # ----------------------------------------------------------------------

#     def _generate_metadata(self, df: DataFrame) -> Dict[str, Any]:
#         """Generate dataset metadata for PySpark DataFrame."""
#         return {
#             "timestamp": datetime.now().isoformat(),
#             "shape": (df.count(), len(df.columns)),
#             "columns": df.columns,
#             "dtypes": {f.name: f.dataType.simpleString() for f in df.schema.fields},
#         }

#     def _generate_data_overview(self, df: DataFrame) -> Dict[str, Any]:
#         """Generate high-level data overview."""
#         total_rows = df.count()
#         total_columns = len(df.columns)

#         missing_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
#         total_missing = sum(missing_counts.values())
#         missing_percentage = (total_missing / (total_rows * total_columns) * 100) if total_rows > 0 else 0

#         duplicate_rows = df.count() - df.dropDuplicates().count()

#         numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["int", "bigint", "double", "float", "decimal"]]
#         categorical_cols = [c for c in df.columns if c not in numeric_cols]

#         return {
#             "total_rows": total_rows,
#             "total_columns": total_columns,
#             "missing_values": total_missing,
#             "missing_percentage": f"{missing_percentage:.2f}%",
#             "duplicate_rows": duplicate_rows,
#             "numeric_columns": len(numeric_cols),
#             "categorical_columns": len(categorical_cols),
#         }

#     def _generate_column_analysis(self, df: DataFrame) -> Dict[str, Any]:
#         """Generate column-level analysis using PySpark ops."""
#         analysis = {}
#         total_rows = df.count()

#         for field in df.schema.fields:
#             col = field.name
#             dtype = field.dataType.simpleString()

#             stats = df.select(
#                 F.count(F.when(F.col(col).isNotNull(), col)).alias("non_null"),
#                 F.count(F.when(F.col(col).isNull(), col)).alias("null_count"),
#             ).collect()[0].asDict()

#             non_null = stats["non_null"]
#             null_count = stats["null_count"]
#             null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

#             unique_count = df.select(col).distinct().count()
#             unique_percentage = (unique_count / total_rows * 100) if total_rows > 0 else 0

#             col_info = {
#                 "dtype": dtype,
#                 "non_null": non_null,
#                 "null_count": null_count,
#                 "null_percentage": f"{null_percentage:.2f}%",
#                 "unique_count": unique_count,
#                 "unique_percentage": f"{unique_percentage:.2f}%",
#             }

#             if dtype in ["int", "bigint", "double", "float", "decimal"]:
#                 agg = df.select(
#                     F.mean(col).alias("mean"),
#                     F.expr(f"percentile_approx({col}, 0.5)").alias("median"),
#                     F.stddev(col).alias("std"),
#                     F.min(col).alias("min"),
#                     F.max(col).alias("max"),
#                 ).collect()[0].asDict()
#                 col_info.update(agg)
#             else:
#                 top_values = df.groupBy(col).count().orderBy(F.desc("count")).limit(5)
#                 col_info["top_values"] = {r[col]: r["count"] for r in top_values.collect() if r[col] is not None}

#             analysis[col] = col_info

#         return analysis

#     def _generate_preprocessing_summary(self, suggestions: Dict = None, snapshots: list = None) -> Dict[str, Any]:
#         """Generate summary of preprocessing steps."""
#         if not suggestions:
#             return {"message": "No preprocessing suggestions available"}

#         summary = {
#             "total_suggestions": len(suggestions),
#             "applied_operations": [],
#             "skipped_columns": [],
#         }

#         for col, suggestion in suggestions.items():
#             if suggestion.get("suggestion") == "skip":
#                 summary["skipped_columns"].append(col)
#             else:
#                 summary["applied_operations"].append(
#                     {
#                         "column": col,
#                         "operation": suggestion.get("suggestion"),
#                         "reason": suggestion.get("reason", ""),
#                     }
#                 )

#         if snapshots:
#             summary["total_snapshots"] = len(snapshots)
#             summary["snapshot_steps"] = [s["step"] for s in snapshots]

#         return summary

#     def _generate_quality_assessment(self, df: DataFrame) -> Dict[str, Any]:
#         """Generate data quality assessment for PySpark DataFrame."""
#         total_rows = df.count()
#         total_cols = len(df.columns)

#         missing_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
#         missing_pct = (sum(missing_counts.values()) / (total_rows * total_cols) * 100) if total_rows > 0 else 0

#         dup_pct = ((df.count() - df.dropDuplicates().count()) / total_rows * 100) if total_rows > 0 else 0

#         quality_score = 100
#         issues = []

#         if missing_pct > 10:
#             quality_score -= 20
#             issues.append(f"High missing values: {missing_pct:.1f}%")
#         elif missing_pct > 5:
#             quality_score -= 10
#             issues.append(f"Moderate missing values: {missing_pct:.1f}%")

#         if dup_pct > 5:
#             quality_score -= 15
#             issues.append(f"High duplicate rate: {dup_pct:.1f}%")
#         elif dup_pct > 1:
#             quality_score -= 5
#             issues.append(f"Moderate duplicate rate: {dup_pct:.1f}%")

#         constant_cols = [c for c in df.columns if df.select(c).distinct().count() <= 1]
#         if constant_cols:
#             quality_score -= 10
#             issues.append(f"Constant columns found: {constant_cols}")

#         return {
#             "quality_score": max(quality_score, 0),
#             "issues": issues,
#             "assessment": "Excellent"
#             if quality_score >= 90
#             else "Good"
#             if quality_score >= 70
#             else "Fair"
#             if quality_score >= 50
#             else "Poor",
#         }

#     def _generate_recommendations(self, df: DataFrame, inspection_results: Dict = None) -> list:
#         """Generate data recommendations for PySpark DataFrame."""
#         recommendations = []

#         numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["int", "bigint", "double", "float", "decimal"]]
#         categorical_cols = [c for c in df.columns if c not in numeric_cols]

#         for c in categorical_cols:
#             uniq = df.select(c).distinct().count()
#             total = df.count()
#             if total > 0 and uniq > total * 0.5:
#                 recommendations.append(f"Consider feature engineering for high-cardinality column: {c}")

#         for c in numeric_cols:
#             stats = df.select(
#                 F.mean(c).alias("mean"),
#                 F.stddev(c).alias("std"),
#                 F.min(c).alias("min"),
#                 F.max(c).alias("max"),
#                 F.skewness(c).alias("skew"),
#             ).collect()[0].asDict()
#             if abs(stats["skew"] or 0) > 2:
#                 recommendations.append(f"Column '{c}' is highly skewed (skewness: {stats['skew']:.2f}). Consider transformation.")

#         if not recommendations:
#             recommendations.append("Dataset appears to be in good condition for analysis.")

#         return recommendations

#     # ----------------------------------------------------------------------
#     # HTML generation (same style as pandas version)
#     # ----------------------------------------------------------------------
#     def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
#         """Generate HTML report content."""
# html = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Data Cleaning Report - {report_data['metadata']['timestamp'][:10]}</title>
#     <style>
#         body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
#         .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
#         h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
#         h2 {{ color: #555; margin-top: 30px; }}
#         h3 {{ color: #666; }}
#         .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
#         .quality-score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
#         .issue {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
#         .recommendation {{ background-color: #d1ecf1; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }}
#         table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
#         th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
#         th {{ background-color: #f8f9fa; font-weight: bold; }}
#         .operation {{ background-color: #e8f5e8; padding: 8px; margin: 5px 0; border-radius: 4px; }}
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>📊 Data Cleaning & Preprocessing Report</h1>
        
#         <div class="metric">
#             <h3>📅 Generated on: {report_data['metadata']['timestamp']}</h3>
#             <h3>📈 Dataset Shape: {report_data['metadata']['shape'][0]:,} rows × {report_data['metadata']['shape'][1]} columns</h3>
#             <h3>💾 Memory Usage: {report_data['metadata']['memory_usage']}</h3>
#         </div>
        
#         <h2>📋 Data Overview</h2>
#         <div class="metric">
#             <p><strong>Total Rows:</strong> {report_data['data_overview']['total_rows']:,}</p>
#             <p><strong>Total Columns:</strong> {report_data['data_overview']['total_columns']}</p>
#             <p><strong>Missing Values:</strong> {report_data['data_overview']['missing_values']:,} ({report_data['data_overview']['missing_percentage']})</p>
#             <p><strong>Duplicate Rows:</strong> {report_data['data_overview']['duplicate_rows']:,}</p>
#             <p><strong>Numeric Columns:</strong> {report_data['data_overview']['numeric_columns']}</p>
#             <p><strong>Categorical Columns:</strong> {report_data['data_overview']['categorical_columns']}</p>
#         </div>
        
#         <h2>🔍 Column Analysis</h2>
#         <table>
#             <tr>
#                 <th>Column</th>
#                 <th>Type</th>
#                 <th>Non-Null</th>
#                 <th>Missing %</th>
#                 <th>Unique Count</th>
#                 <th>Summary</th>
#             </tr>
# """
		
# 		for col, info in report_data['column_analysis'].items():
# 			summary = ""
# 			if 'mean' in info and info['mean'] is not None:
# 				summary = f"Mean: {info['mean']:.2f}, Range: {info['min']:.2f} - {info['max']:.2f}"
# 			elif 'top_values' in info:
# 				top_val = list(info['top_values'].keys())[0]
# 				summary = f"Top value: {top_val} ({info['top_values'][top_val]} occurrences)"
			
# 			html += f"""
#             <tr>
#                 <td><strong>{col}</strong></td>
#                 <td>{info['dtype']}</td>
#                 <td>{info['non_null']:,}</td>
#                 <td>{info['null_percentage']}</td>
#                 <td>{info['unique_count']:,}</td>
#                 <td>{summary}</td>
#             </tr>
# """
		
# 		html += """
#         </table>
        
#         <h2>⚙️ Preprocessing Summary</h2>
#         <div class="metric">
#             <p><strong>Total Suggestions:</strong> {total_suggestions}</p>
#             <p><strong>Applied Operations:</strong> {applied_count}</p>
#             <p><strong>Skipped Columns:</strong> {skipped_count}</p>
#         </div>
        
#         <h3>Applied Operations:</h3>
# """.format(
# 			total_suggestions=report_data['preprocessing_summary'].get('total_suggestions', 0),
# 			applied_count=len(report_data['preprocessing_summary'].get('applied_operations', [])),
# 			skipped_count=len(report_data['preprocessing_summary'].get('skipped_columns', []))
# 		)
		
# 		for op in report_data['preprocessing_summary'].get('applied_operations', []):
# 			html += f"""
#         <div class="operation">
#             <strong>{op['column']}:</strong> {op['operation']} - {op['reason']}
#         </div>
# """
		
# 		html += f"""
#         <h2>🎯 Data Quality Assessment</h2>
#         <div class="metric">
#             <div class="quality-score">Quality Score: {report_data['quality_assessment']['quality_score']}/100</div>
#             <p><strong>Assessment:</strong> {report_data['quality_assessment']['assessment']}</p>
#         </div>
# """
		
# 		if report_data['quality_assessment']['issues']:
# 			html += "<h3>⚠️ Issues Found:</h3>"
# 			for issue in report_data['quality_assessment']['issues']:
# 				html += f"<div class='issue'>{issue}</div>"
		
# 		html += """
#         <h2>💡 Recommendations</h2>
# """
		
# 		for rec in report_data['recommendations']:
# 			html += f"<div class='recommendation'>{rec}</div>"
		
# 		html += """
#         <hr style="margin: 40px 0;">
#         <p style="text-align: center; color: #666;">
#             Report generated by AI Data Cleaner | 
#             <a href="#" onclick="window.print()">Print Report</a>
#         </p>
#     </div>
# </body>
# </html>
# """
		
#        return html
