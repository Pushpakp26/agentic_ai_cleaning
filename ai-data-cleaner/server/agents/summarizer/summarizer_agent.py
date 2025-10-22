from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    from ..base_agent import BaseAgent
    from ...config import PROCESSED_DIR
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from config import PROCESSED_DIR
    from utils.logger import get_logger


logger = get_logger(__name__)


class SummarizerAgent(BaseAgent):
	"""Agent for generating summary reports."""
	
	def __init__(self):
		super().__init__("SummarizerAgent")
		self.output_dir = PROCESSED_DIR
		self.output_dir.mkdir(exist_ok=True)
	
	def process(self, df: pd.DataFrame, inspection_results: Dict = None, 
				suggestions: Dict = None, snapshots: list = None, applied_operations: list = None, **kwargs) -> Dict[str, Any]:
		"""Generate a comprehensive summary report."""
		
		# Generate report data
		report_data = {
			"metadata": self._generate_metadata(df),
			"data_overview": self._generate_data_overview(df),
			"column_analysis": self._generate_column_analysis(df),
			"preprocessing_summary": self._generate_preprocessing_summary(suggestions, snapshots, applied_operations),
			"quality_assessment": self._generate_quality_assessment(df),
			"recommendations": self._generate_recommendations(df, inspection_results)
		}
		
		# Generate HTML report
		html_content = self._generate_html_report(report_data)
		
		# Save report
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		report_path = self.output_dir / f"report_{timestamp}.html"
		report_path.write_text(html_content, encoding='utf-8')
		
		logger.info(f"Generated summary report: {report_path}")
		
		return {
			"report_path": str(report_path),
			"report_data": report_data,
			"html_content": html_content
		}
	
	def _generate_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
		"""Generate dataset metadata."""
		return {
			"timestamp": datetime.now().isoformat(),
			"shape": df.shape,
			"memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
			"columns": list(df.columns),
			"dtypes": df.dtypes.astype(str).to_dict()
		}
	
	def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
		"""Generate high-level data overview."""
		return {
			"total_rows": len(df),
			"total_columns": len(df.columns),
			"missing_values": int(df.isnull().sum().sum()),
			"missing_percentage": f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%",
			"duplicate_rows": int(df.duplicated().sum()),
			"numeric_columns": len(df.select_dtypes(include=['number']).columns),
			"categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns)
		}
	
	def _generate_column_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
		"""Generate detailed column analysis."""
		analysis = {}
		
		for col in df.columns:
			series = df[col]
			col_info = {
				"dtype": str(series.dtype),
				"non_null": int(series.notna().sum()),
				"null_count": int(series.isna().sum()),
				"null_percentage": f"{(series.isna().sum() / len(series) * 100):.2f}%",
				"unique_count": int(series.nunique()),
				"unique_percentage": f"{(series.nunique() / len(series) * 100):.2f}%"
			}
			
			if pd.api.types.is_numeric_dtype(series):
				col_info.update({
					"mean": float(series.mean()) if not series.isna().all() else None,
					"median": float(series.median()) if not series.isna().all() else None,
					"std": float(series.std()) if not series.isna().all() else None,
					"min": float(series.min()) if not series.isna().all() else None,
					"max": float(series.max()) if not series.isna().all() else None
				})
			else:
				# Top values for categorical columns
				top_values = series.value_counts().head(5)
				col_info["top_values"] = top_values.to_dict()
			
			analysis[col] = col_info
		
		return analysis
	
	def _generate_preprocessing_summary(self, suggestions: Dict = None, snapshots: list = None, applied_operations: list = None) -> Dict[str, Any]:
		"""Generate summary of preprocessing steps."""
		if not suggestions and not applied_operations:
			return {"message": "No preprocessing details available", "total_suggestions": 0, "applied_operations": [], "skipped_columns": []}
		
		summary = {
			"total_suggestions": len(suggestions) if suggestions else 0,
			"applied_operations": [],
			"skipped_columns": []
		}
		# Prefer actual applied operations if provided
		if applied_operations:
			summary["applied_operations"] = list(applied_operations)
		else:
			# Fallback: derive from suggestions (legacy behavior)
			for col, suggestion in (suggestions or {}).items():
				# Handle both single suggestion (dict) and multiple suggestions (list)
				if not isinstance(suggestion, list):
					suggestion = [suggestion]
				for action in suggestion:
					if action.get("suggestion") == "skip":
						summary["skipped_columns"].append(col)
					else:
						summary["applied_operations"].append({
							"column": col,
							"operation": action.get("suggestion"),
							"reason": action.get("reason", "")
						})
		
		if snapshots:
			summary["total_snapshots"] = len(snapshots)
			summary["snapshot_steps"] = [s["step"] for s in snapshots]
		
		return summary
	
	def _generate_quality_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
		"""Generate data quality assessment."""
		quality_score = 100
		issues = []
		
		# Check for missing values
		missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
		if missing_pct > 10:
			quality_score -= 20
			issues.append(f"High missing values: {missing_pct:.1f}%")
		elif missing_pct > 5:
			quality_score -= 10
			issues.append(f"Moderate missing values: {missing_pct:.1f}%")
		
		# Check for duplicates
		dup_pct = (df.duplicated().sum() / len(df) * 100)
		if dup_pct > 5:
			quality_score -= 15
			issues.append(f"High duplicate rate: {dup_pct:.1f}%")
		elif dup_pct > 1:
			quality_score -= 5
			issues.append(f"Moderate duplicate rate: {dup_pct:.1f}%")
		
		# Check for constant columns
		constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
		if constant_cols:
			quality_score -= 10
			issues.append(f"Constant columns found: {constant_cols}")
		
		return {
			"quality_score": max(quality_score, 0),
			"issues": issues,
			"assessment": "Excellent" if quality_score >= 90 else 
						  "Good" if quality_score >= 70 else
						  "Fair" if quality_score >= 50 else "Poor"
		}
	
	def _generate_recommendations(self, df: pd.DataFrame, inspection_results: Dict = None) -> list:
		"""Generate recommendations for further analysis."""
		recommendations = []
		
		# Check for high-cardinality categorical columns
		high_card_cols = [col for col in df.select_dtypes(include=['object']).columns 
						 if df[col].nunique() > len(df) * 0.5]
		if high_card_cols:
			recommendations.append(f"Consider feature engineering for high-cardinality columns: {high_card_cols}")
		
		# Check for skewed numeric columns
		numeric_cols = df.select_dtypes(include=['number']).columns
		for col in numeric_cols:
			skewness = df[col].skew()
			if abs(skewness) > 2:
				recommendations.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f}). Consider transformation.")
		
		# Check for potential outliers
		for col in numeric_cols:
			Q1, Q3 = df[col].quantile([0.25, 0.75])
			IQR = Q3 - Q1
			outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
			if len(outliers) > len(df) * 0.05:
				recommendations.append(f"Column '{col}' has potential outliers ({len(outliers)} rows). Consider investigation.")
		
		if not recommendations:
			recommendations.append("Dataset appears to be in good condition for analysis.")
		
		return recommendations
	
	def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
		"""Generate HTML report content."""
		html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Cleaning Report - {report_data['metadata']['timestamp'][:10]}</title>
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
            <h3>📅 Generated on: {report_data['metadata']['timestamp']}</h3>
            <h3>📈 Dataset Shape: {report_data['metadata']['shape'][0]:,} rows × {report_data['metadata']['shape'][1]} columns</h3>
            <h3>💾 Memory Usage: {report_data['metadata']['memory_usage']}</h3>
        </div>
        
        <h2>📋 Data Overview</h2>
        <div class="metric">
            <p><strong>Total Rows:</strong> {report_data['data_overview']['total_rows']:,}</p>
            <p><strong>Total Columns:</strong> {report_data['data_overview']['total_columns']}</p>
            <p><strong>Missing Values:</strong> {report_data['data_overview']['missing_values']:,} ({report_data['data_overview']['missing_percentage']})</p>
            <p><strong>Duplicate Rows:</strong> {report_data['data_overview']['duplicate_rows']:,}</p>
            <p><strong>Numeric Columns:</strong> {report_data['data_overview']['numeric_columns']}</p>
            <p><strong>Categorical Columns:</strong> {report_data['data_overview']['categorical_columns']}</p>
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
		
		for col, info in report_data['column_analysis'].items():
			summary = ""
			if 'mean' in info and info['mean'] is not None:
				summary = f"Mean: {info['mean']:.2f}, Range: {info['min']:.2f} - {info['max']:.2f}"
			elif 'top_values' in info:
				top_val = list(info['top_values'].keys())[0]
				summary = f"Top value: {top_val} ({info['top_values'][top_val]} occurrences)"
			
			html += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{info['dtype']}</td>
                <td>{info['non_null']:,}</td>
                <td>{info['null_percentage']}</td>
                <td>{info['unique_count']:,}</td>
                <td>{summary}</td>
            </tr>
"""
		
		html += """
        </table>
        
        <h2>⚙️ Preprocessing Summary</h2>
        <div class="metric">
            <p><strong>Total Suggestions:</strong> {total_suggestions}</p>
            <p><strong>Applied Operations:</strong> {applied_count}</p>
            <p><strong>Skipped Columns:</strong> {skipped_count}</p>
        </div>
        
        <h3>Applied Operations:</h3>
""".format(
		total_suggestions=report_data['preprocessing_summary'].get('total_suggestions', 0),
		applied_count=len(report_data['preprocessing_summary'].get('applied_operations', [])),
		skipped_count=len(report_data['preprocessing_summary'].get('skipped_columns', []))
	)
	
		for op in report_data['preprocessing_summary'].get('applied_operations', []):
			html += f"""
			<div class="operation">
				<strong>{op.get('column', '')}:</strong> {op.get('operation', '')} - {op.get('reason', '')}
			</div>
			"""
		
		html += f"""
		<h2>🎯 Data Quality Assessment</h2>
		<div class="metric">
			<div class="quality-score">Quality Score: {report_data['quality_assessment']['quality_score']}/100</div>
			<p><strong>Assessment:</strong> {report_data['quality_assessment']['assessment']}</p>
		</div>
		"""
		
		if report_data['quality_assessment']['issues']:
			html += "<h3>⚠️ Issues Found:</h3>"
			for issue in report_data['quality_assessment']['issues']:
				html += f"<div class='issue'>{issue}</div>"

		html += """
		<h2>💡 Recommendations</h2>
		"""

		for rec in report_data['recommendations']:
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