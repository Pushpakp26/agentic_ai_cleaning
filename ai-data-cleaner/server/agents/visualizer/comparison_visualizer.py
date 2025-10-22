"""
Comparison Visualizer Agent
Generates before/after visualizations and statistical comparisons for numerical columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import base64
from io import BytesIO

try:
    from ...utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonVisualizer:
    """Generate before/after comparison visualizations and reports."""
    
    def __init__(self):
        self.style_config()
    
    def style_config(self):
        """Configure matplotlib style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def process(self, 
                original_df: pd.DataFrame, 
                processed_df: pd.DataFrame,
                original_column_types: Dict[str, str],
                session_id: str) -> Dict[str, Any]:
        """
        Generate comparison visualizations and report.
        
        Args:
            original_df: Original dataset before processing
            processed_df: Processed dataset after pipeline
            original_column_types: Dictionary mapping column names to types
            session_id: Session identifier for file naming
            
        Returns:
            Dictionary with visualization paths and statistics
        """
        logger.info("Starting before/after comparison visualization...")
        
        # Find common numerical columns
        common_numerical_cols = self._find_common_numerical_columns(
            original_df, processed_df, original_column_types
        )
        
        logger.info(f"Found {len(common_numerical_cols)} common numerical columns for comparison")
        
        if not common_numerical_cols:
            logger.warning("No common numerical columns found for comparison")
            return {"visualizations": {}, "statistics": {}}
        
        # Generate visualizations for each column
        visualizations = {}
        statistics = {}
        
        for col in common_numerical_cols:
            try:
                # Generate comparison plots
                viz_data = self._generate_column_comparison(
                    original_df, processed_df, col
                )
                visualizations[col] = viz_data
                
                # Calculate statistics
                stats = self._calculate_column_statistics(
                    original_df, processed_df, col
                )
                statistics[col] = stats
                
            except Exception as e:
                logger.error(f"Failed to generate comparison for column '{col}': {e}")
        
        # Generate HTML report
        report_html = self._generate_html_report(
            visualizations, statistics, session_id
        )
        
        return {
            "visualizations": visualizations,
            "statistics": statistics,
            "report_html": report_html
        }
    
    def _find_common_numerical_columns(self,
                                       original_df: pd.DataFrame,
                                       processed_df: pd.DataFrame,
                                       original_column_types: Dict[str, str]) -> List[str]:
        """Find columns that exist in both datasets and were originally numerical."""
        common_cols = set(original_df.columns) & set(processed_df.columns)
        
        numerical_cols = []
        for col in common_cols:
            # Check if it was originally numerical
            if original_column_types.get(col) == 'numerical':
                # Verify it's still numerical in processed df
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    numerical_cols.append(col)
        
        return sorted(numerical_cols)
    
    def _generate_column_comparison(self,
                                    original_df: pd.DataFrame,
                                    processed_df: pd.DataFrame,
                                    col: str) -> Dict[str, str]:
        """Generate before/after comparison plots for a column."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Before vs After Comparison: {col}', fontsize=16, fontweight='bold')
        
        original_data = original_df[col].dropna()
        processed_data = processed_df[col].dropna()
        
        # Row 1: Original (Before)
        # Distribution
        axes[0, 0].hist(original_data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Before: Distribution')
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(original_data, vert=True)
        axes[0, 1].set_title('Before: Box Plot')
        axes[0, 1].set_ylabel(col)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(original_data, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Before: Q-Q Plot')
        
        # Row 2: Processed (After)
        # Distribution
        axes[1, 0].hist(processed_data, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('After: Distribution')
        axes[1, 0].set_xlabel(col)
        axes[1, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[1, 1].boxplot(processed_data, vert=True)
        axes[1, 1].set_title('After: Box Plot')
        axes[1, 1].set_ylabel(col)
        
        # Q-Q plot
        stats.probplot(processed_data, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('After: Q-Q Plot')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    def _calculate_column_statistics(self,
                                     original_df: pd.DataFrame,
                                     processed_df: pd.DataFrame,
                                     col: str) -> Dict[str, Any]:
        """Calculate before/after statistics for a column."""
        
        original_data = original_df[col]
        processed_data = processed_df[col]
        
        stats = {
            "before": {
                "count": int(original_data.count()),
                "missing": int(original_data.isna().sum()),
                "missing_pct": float(original_data.isna().sum() / len(original_data) * 100),
                "mean": float(original_data.mean()) if original_data.count() > 0 else None,
                "median": float(original_data.median()) if original_data.count() > 0 else None,
                "std": float(original_data.std()) if original_data.count() > 0 else None,
                "min": float(original_data.min()) if original_data.count() > 0 else None,
                "max": float(original_data.max()) if original_data.count() > 0 else None,
                "range": float(original_data.max() - original_data.min()) if original_data.count() > 0 else None,
                "skewness": float(original_data.skew()) if original_data.count() > 0 else None,
                "kurtosis": float(original_data.kurtosis()) if original_data.count() > 0 else None,
            },
            "after": {
                "count": int(processed_data.count()),
                "missing": int(processed_data.isna().sum()),
                "missing_pct": float(processed_data.isna().sum() / len(processed_data) * 100),
                "mean": float(processed_data.mean()) if processed_data.count() > 0 else None,
                "median": float(processed_data.median()) if processed_data.count() > 0 else None,
                "std": float(processed_data.std()) if processed_data.count() > 0 else None,
                "min": float(processed_data.min()) if processed_data.count() > 0 else None,
                "max": float(processed_data.max()) if processed_data.count() > 0 else None,
                "range": float(processed_data.max() - processed_data.min()) if processed_data.count() > 0 else None,
                "skewness": float(processed_data.skew()) if processed_data.count() > 0 else None,
                "kurtosis": float(processed_data.kurtosis()) if processed_data.count() > 0 else None,
            }
        }
        
        # Calculate improvements (with meaningful thresholds)
        before_skew = abs(stats["before"]["skewness"] or 0)
        after_skew = abs(stats["after"]["skewness"] or 0)
        skew_reduction = before_skew - after_skew
        
        before_std = stats["before"]["std"] or 0
        after_std = stats["after"]["std"] or 0
        std_reduction = before_std - after_std
        
        stats["improvements"] = {
            "missing_reduced": stats["before"]["missing"] - stats["after"]["missing"],
            "skewness_improved": skew_reduction > 0.01,  # At least 0.01 reduction to count as improvement
            "outliers_handled": std_reduction > 0.01 if before_std > 0 and after_std > 0 else False
        }
        
        return stats
    
    def _generate_html_report(self,
                             visualizations: Dict[str, str],
                             statistics: Dict[str, Dict],
                             session_id: str) -> str:
        """Generate HTML report with visualizations and statistics."""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Before vs After Comparison Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .header {{
            background: white;
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 5px;
        }}
        .header-info {{
            color: #666;
            font-size: 0.9em;
        }}
        .navbar {{
            background: white;
            padding: 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 80px;
            z-index: 99;
            overflow-x: auto;
            white-space: nowrap;
        }}
        .navbar::-webkit-scrollbar {{
            height: 6px;
        }}
        .navbar::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        .navbar::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 3px;
        }}
        .nav-tabs {{
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
        }}
        .nav-tab {{
            padding: 15px 25px;
            cursor: pointer;
            background: #f8f9fa;
            border-right: 1px solid #ddd;
            transition: all 0.3s ease;
            font-weight: 500;
            color: #555;
        }}
        .nav-tab:hover {{
            background: #e9ecef;
            color: #2c3e50;
        }}
        .nav-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .column-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .column-header h2 {{
            font-size: 1.8em;
            margin-bottom: 10px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .stats-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .improvement {{
            color: #27ae60;
            font-weight: bold;
        }}
        .degradation {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .summary-box {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            color: #2c3e50;
            font-size: 1.1em;
        }}
    </style>
    <script>
        function showTab(columnName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById('tab-' + columnName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
        
        window.onload = function() {{
            // Show first tab by default
            const firstTab = document.querySelector('.nav-tab');
            if (firstTab) {{
                firstTab.click();
            }}
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>📊 Before vs After Comparison Report</h1>
        <div class="header-info">
            <strong>Session ID:</strong> {session_id} | 
            <strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            <strong>Columns Analyzed:</strong> {len(visualizations)} numerical columns
        </div>
    </div>
    
    <div class="navbar">
        <ul class="nav-tabs">
"""
        
        # Generate navigation tabs
        for col in visualizations.keys():
            html += f'            <li class="nav-tab" onclick="showTab(\'{col}\')">{col}</li>\n'
        
        html += """
        </ul>
    </div>
    
    <div class="container">
"""
        
        # Generate tab content for each column
        for col, viz_data in visualizations.items():
            stats = statistics.get(col, {})
            before = stats.get("before", {})
            after = stats.get("after", {})
            improvements = stats.get("improvements", {})
            
            html += f"""
        <div id="tab-{col}" class="tab-content">
            <div class="column-header">
                <h2>{col}</h2>
                <p>Detailed before/after comparison and statistical analysis</p>
            </div>
            
            <div class="summary-box">
                <h3>Summary</h3>
                <div class="metric">
                    <span class="metric-label">Missing Values:</span>
                    <span class="metric-value">{before.get('missing', 0)} → {after.get('missing', 0)}</span>
                    {'<span class="improvement">✓ Improved</span>' if improvements.get('missing_reduced', 0) > 0 else ''}
                </div>
                <div class="metric">
                    <span class="metric-label">Skewness:</span>
                    <span class="metric-value">{before.get('skewness', 0):.3f} → {after.get('skewness', 0):.3f}</span>
                    {'<span class="improvement">✓ Improved</span>' if improvements.get('skewness_improved') else ''}
                </div>
            </div>
            
            <div class="visualization">
                <img src="{viz_data}" alt="Comparison for {col}">
            </div>
            
            <h3>Detailed Statistics</h3>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Before</th>
                        <th>After</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Count</td>
                        <td>{before.get('count', 0)}</td>
                        <td>{after.get('count', 0)}</td>
                        <td>{after.get('count', 0) - before.get('count', 0):+d}</td>
                    </tr>
                    <tr>
                        <td>Missing Values</td>
                        <td>{before.get('missing', 0)} ({before.get('missing_pct', 0):.2f}%)</td>
                        <td>{after.get('missing', 0)} ({after.get('missing_pct', 0):.2f}%)</td>
                        <td class="{'improvement' if improvements.get('missing_reduced', 0) > 0 else ''}">{improvements.get('missing_reduced', 0):+d}</td>
                    </tr>
                    <tr>
                        <td>Mean</td>
                        <td>{before.get('mean', 0):.4f}</td>
                        <td>{after.get('mean', 0):.4f}</td>
                        <td>{(after.get('mean', 0) - before.get('mean', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Median</td>
                        <td>{before.get('median', 0):.4f}</td>
                        <td>{after.get('median', 0):.4f}</td>
                        <td>{(after.get('median', 0) - before.get('median', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Std Dev</td>
                        <td>{before.get('std', 0):.4f}</td>
                        <td>{after.get('std', 0):.4f}</td>
                        <td>{(after.get('std', 0) - before.get('std', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Min</td>
                        <td>{before.get('min', 0):.4f}</td>
                        <td>{after.get('min', 0):.4f}</td>
                        <td>{(after.get('min', 0) - before.get('min', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Max</td>
                        <td>{before.get('max', 0):.4f}</td>
                        <td>{after.get('max', 0):.4f}</td>
                        <td>{(after.get('max', 0) - before.get('max', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Range</td>
                        <td>{before.get('range', 0):.4f}</td>
                        <td>{after.get('range', 0):.4f}</td>
                        <td>{(after.get('range', 0) - before.get('range', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Skewness</td>
                        <td>{before.get('skewness', 0):.4f}</td>
                        <td>{after.get('skewness', 0):.4f}</td>
                        <td class="{'improvement' if improvements.get('skewness_improved') else ''}">{(after.get('skewness', 0) - before.get('skewness', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Kurtosis</td>
                        <td>{before.get('kurtosis', 0):.4f}</td>
                        <td>{after.get('kurtosis', 0):.4f}</td>
                        <td>{(after.get('kurtosis', 0) - before.get('kurtosis', 0)):.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""
        
        # Close the container div and body
        html += """
    </div>
</body>
</html>
"""
        
        return html
