import base64
import io
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

try:
    from ..base_agent import BaseAgent
    from ...config import PROCESSED_DIR
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from config import PROCESSED_DIR
    from utils.logger import get_logger


logger = get_logger(__name__)


class VisualizerAgent(BaseAgent):
	"""Agent for generating data visualizations."""
	
	def __init__(self):
		super().__init__("VisualizerAgent")
		plt.style.use('seaborn-v0_8')
		self.output_dir = PROCESSED_DIR / "visualizations"
		self.output_dir.mkdir(exist_ok=True)
	
	def process(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
		"""Generate visualizations for the dataframe."""
		visualizations = {}
		
		try:
			logger.info(f"Generating visualizations for dataframe with shape {df.shape}")
			
			# Generate numeric column visualizations
			try:
				numeric_cols = df.select_dtypes(include=['number']).columns
				logger.info(f"Processing {len(numeric_cols)} numeric columns")
				for col in numeric_cols:
					try:
						visualizations[f"{col}_histogram"] = self._create_histogram(df, col)
					except Exception as e:
						logger.warning(f"Failed to create histogram for '{col}': {e}")
					try:
						visualizations[f"{col}_boxplot"] = self._create_boxplot(df, col)
					except Exception as e:
						logger.warning(f"Failed to create boxplot for '{col}': {e}")
			except Exception as e:
				logger.warning(f"Failed to process numeric columns: {e}")
			
			# Generate categorical column visualizations
			try:
				categorical_cols = df.select_dtypes(include=['object', 'category']).columns
				logger.info(f"Processing {len(categorical_cols)} categorical columns")
				for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
					try:
						visualizations[f"{col}_bar"] = self._create_bar_chart(df, col)
					except Exception as e:
						logger.warning(f"Failed to create bar chart for '{col}': {e}")
			except Exception as e:
				logger.warning(f"Failed to process categorical columns: {e}")
			
			# Generate correlation heatmap for numeric columns
			try:
				numeric_cols = df.select_dtypes(include=['number']).columns
				if len(numeric_cols) > 1:
					visualizations["correlation_heatmap"] = self._create_correlation_heatmap(df, numeric_cols)
			except Exception as e:
				logger.warning(f"Failed to create correlation heatmap: {e}")
			
			# Generate missing values heatmap
			try:
				if df.isnull().any().any():
					visualizations["missing_values"] = self._create_missing_values_plot(df)
			except Exception as e:
				logger.warning(f"Failed to create missing values plot: {e}")
			
			logger.info(f"Generated {len(visualizations)} visualizations successfully")
			return visualizations
			
		except Exception as e:
			logger.error(f"Visualization generation failed: {e}", exc_info=True)
			return {}
	
	def _create_histogram(self, df: pd.DataFrame, column: str) -> str:
		"""Create histogram for numeric column."""
		fig, ax = plt.subplots(figsize=(8, 6))
		
		# Remove missing values for plotting
		data = df[column].dropna()
		
		ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
		ax.set_title(f'Distribution of {column}')
		ax.set_xlabel(column)
		ax.set_ylabel('Frequency')
		ax.grid(True, alpha=0.3)
		
		return self._figure_to_base64(fig)
	
	def _create_boxplot(self, df: pd.DataFrame, column: str) -> str:
		"""Create boxplot for numeric column."""
		fig, ax = plt.subplots(figsize=(8, 6))
		
		# Remove missing values for plotting
		data = df[column].dropna()
		
		ax.boxplot(data)
		ax.set_title(f'Boxplot of {column}')
		ax.set_ylabel(column)
		ax.grid(True, alpha=0.3)
		
		return self._figure_to_base64(fig)
	
	def _create_bar_chart(self, df: pd.DataFrame, column: str) -> str:
		"""Create bar chart for categorical column."""
		fig, ax = plt.subplots(figsize=(10, 6))
		
		try:
			# Convert to string and handle missing values
			series = df[column].astype(str).fillna('<Missing>')
			
			# Check for concatenated data (very long strings)
			max_length = series.str.len().max()
			if max_length > 100:
				logger.warning(f"Detected concatenated data in column '{column}' (max length: {max_length}). Cleaning...")
				
				# Extract first categorical value from concatenated string
				def extract_first_category(value):
					if pd.isna(value) or value == '' or value == '<Missing>':
						return value
					value_str = str(value)
					# Look for common categorical patterns
					for pattern in ['Male', 'Female', 'male', 'female', 'M', 'F', 'Yes', 'No', 'True', 'False', 'High', 'Medium', 'Low']:
						if pattern.lower() in value_str.lower():
							return pattern
					# If no pattern found, take first 20 characters
					return value_str[:20] if len(value_str) > 20 else value_str
				
				series = series.apply(extract_first_category)
				logger.info(f"Cleaned concatenated data in column '{column}'")
			
			# Get value counts and limit to top 20 categories
			value_counts = series.value_counts().head(20)
			
			# Create bar chart
			value_counts.plot(kind='bar', ax=ax)
			ax.set_title(f'Top Categories in {column}')
			ax.set_xlabel(column)
			ax.set_ylabel('Count')
			ax.tick_params(axis='x', rotation=45)
			ax.grid(True, alpha=0.3)
			
			return self._figure_to_base64(fig)
			
		except Exception as e:
			logger.error(f"Error creating bar chart for column '{column}': {e}")
			plt.close(fig)
			raise
	
	def _create_correlation_heatmap(self, df: pd.DataFrame, columns) -> str:
		"""Create correlation heatmap for numeric columns."""
		fig, ax = plt.subplots(figsize=(10, 8))
		
		# Calculate correlation matrix
		corr_matrix = df[columns].corr()
		
		# Create heatmap
		sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
					square=True, ax=ax, fmt='.2f')
		ax.set_title('Correlation Heatmap')
		
		return self._figure_to_base64(fig)
	
	def _create_missing_values_plot(self, df: pd.DataFrame) -> str:
		"""Create missing values visualization."""
		fig, ax = plt.subplots(figsize=(12, 8))
		
		# Create missing data matrix
		missing_data = df.isnull()
		
		# Plot missing values heatmap (sample if too large)
		if len(df) > 1000:
			sample_df = df.sample(1000, random_state=42)
			missing_data = sample_df.isnull()
			title = "Missing Values Pattern (Sample of 1000 rows)"
		else:
			title = "Missing Values Pattern"
		
		sns.heatmap(missing_data.T, cbar=True, yticklabels=True, ax=ax)
		ax.set_title(title)
		ax.set_xlabel('Row Index')
		ax.set_ylabel('Columns')
		
		return self._figure_to_base64(fig)
	
	def _figure_to_base64(self, fig: Figure) -> str:
		"""Convert matplotlib figure to base64 string."""
		buffer = io.BytesIO()
		fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
		buffer.seek(0)
		
		image_base64 = base64.b64encode(buffer.read()).decode()
		plt.close(fig)  # Close figure to free memory
		
		return f"data:image/png;base64,{image_base64}"
