# what is remainig ?
# __init__
# ❌
# Only agent objects created. No Spark action yet.


# _inspect_data
# ❌ Potential issue: GeminiInspectorAgent() is a Pandas agent. If self.current_df is PySpark, it may fail.


# _generate_visualizations
# ❌ Problem: VisualizerAgent is Pandas-only

# 8️⃣ _create_summary_report
# ❌ Problem: SummarizerAgent is Pandas-only.

import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

import pandas as pd
from pyspark.sql.types import (IntegerType, LongType, FloatType, DoubleType, 
                                DecimalType, ShortType, ByteType, NumericType)

try:
    from .inspector.gemini_inspector_agent import GeminiInspectorAgent    #pandas inspector
    from .inspector.gemini_inspector_pyspark_agent import GeminiInspectorAgentSpark
    
    from .inspector.nlp_utils import is_nlp_column
    #dedup
    from .dedup.dedup_agent import DedupAgent
    from .dedup.dedup_pyspark_agent import DedupAgentSpark
    #drop constant features
    from .drop_constant_features.drop_constant_features_agent import DropConstantFeaturesAgent
    from .drop_constant_features.drop_constant_pyspark_agent import DropConstantFeaturesAgentPyspark
    #fix infinite values
    from .fix_infinite_values.fix_infinite_agent import FixInfiniteAgent
    from .fix_infinite_values.fix_infinite_pyspark_agent import FixInfiniteValuesAgentPyspark
    #handle datetime
    from .handle_datetime.datetime_agent import HandleDatetimeAgent
    from .handle_datetime.datetime_pyspark_agent import HandleDatetimeAgentPyspark
    #handle skewness
    from .handleSkewness.handle_skewness import HandleSkewnessAgent
    from .handleSkewness.handle_skewness_pyspark_agent import HandleSkewnessAgentPyspark    
    #imputer
    from .imputers.mean_median_imputer import MeanMedianImputerAgent
    from .imputers.imputer_pyspark_agent import ImputersAgent
    #lemmatize
    from .lemmatize.lemmatize_text_agent import LemmatizeTextAgent
    from .lemmatize.lemmatize_text_pyspark_agent import LemmatizationAgentPyspark
    #normalization    
    from .normalization.normalize_agent import NormalizeAgent
    from .normalization.normalize_pyspark_agent import NormalizationAgentPyspark
    #remove stopwords
    from .remove_stopwards.remove_stopwords_agent import RemoveStopwordsAgent
    from .remove_stopwards.remove_stopwords_pyspark_agent import RemoveStopwordsAgentPyspark
    #scaling
    from .scaling.scaler_agent import ScalerAgent
    from .scaling.scaler_pyspark_agent import ScalingAgentPyspark
    #encoding
    from .encoding.encoding_agent import EncodingAgent
    from .encoding.encoding_pyspark_agent import EncodingAgentPyspark
    #summarizer
    from .summarizer.summarizer_agent import SummarizerAgent
    from .summarizer.summarize_sparkagent import SummarizerAgentSpark
    #text
    from .text.text_preprocessing_agent import TextPreprocessingAgent
    from .text.text_cleaning_pyspark_agent import TextCleaningPysparkAgent
    #tfidf
    from .tfidf.tfidf_agent import TFIDFAgent
    from .tfidf.tfidf_pyspark_agent import TfidfPsparkAgent
    #visualizer
    from .visualizer.visualizer_agent import VisualizerAgent
    from .visualizer.visualizer_sparkagent import VisualizerAgentSpark
    from .visualizer.comparison_visualizer import ComparisonVisualizer
    from ..config import is_big_data, SNAPSHOT_DIR, PROCESSED_DIR
    from ..utils.file_handler import read_pandas, write_pandas, detect_file_kind
    from ..utils.spark_session import get_spark, release_spark, get_spark_info
    from ..utils.logger import get_logger
    
    
except ImportError:
    # Inspector / Utility
    from agents.inspector.gemini_inspector_agent import GeminiInspectorAgent
    from agents.inspector.gemini_inspector_pyspark_agent import GeminiInspectorAgentSpark
    from agents.inspector.nlp_utils import is_nlp_column

    # Dedup
    from agents.dedup.dedup_agent import DedupAgent
    from agents.dedup.dedup_pyspark_agent import DedupAgentSpark

    # Drop constant features
    from agents.drop_constant_features.drop_constant_features_agent import DropConstantFeaturesAgent
    from agents.drop_constant_features.drop_constant_pyspark_agent import DropConstantFeaturesAgentPyspark

    # Fix infinite values
    from agents.fix_infinite_values.fix_infinite_agent import FixInfiniteAgent
    from agents.fix_infinite_values.fix_infinite_pyspark_agent import FixInfiniteValuesAgentPyspark

    # Handle datetime
    from agents.handle_datetime.datetime_agent import HandleDatetimeAgent
    from agents.handle_datetime.datetime_pyspark_agent import HandleDatetimeAgentPyspark

    # Handle skewness
    from agents.handleSkewness.handle_skewness import HandleSkewnessAgent
    from agents.handleSkewness.handle_skewness_pyspark_agent import HandleSkewnessAgentPyspark

    # Imputers
    from agents.imputers.mean_median_imputer import MeanMedianImputerAgent
    from agents.imputers.imputer_pyspark_agent import ImputersAgent

    # Lemmatize
    from agents.lemmatize.lemmatize_text_agent import LemmatizeTextAgent
    from agents.lemmatize.lemmatize_text_pyspark_agent import LemmatizationAgentPyspark

    # Normalization
    from agents.normalization.normalize_agent import NormalizeAgent
    from agents.normalization.normalize_pyspark_agent import NormalizationAgentPyspark

    # Remove stopwords
    from agents.remove_stopwards.remove_stopwords_agent import RemoveStopwordsAgent
    from agents.remove_stopwards.remove_stopwords_pyspark_agent import RemoveStopwordsAgentPyspark

    # Scaling
    from agents.scaling.scaler_agent import ScalerAgent
    from agents.scaling.scaler_pyspark_agent import ScalingAgentPyspark

    # Encoding
    from agents.encoding.encoding_agent import EncodingAgent
    from agents.encoding.encoding_pyspark_agent import EncodingAgentPyspark

    # Summarizer
    from agents.summarizer.summarizer_agent import SummarizerAgent
    from agents.summarizer.summarize_sparkagent import SummarizerAgentSpark

    # Text
    from agents.text.text_preprocessing_agent import TextPreprocessingAgent
    from agents.text.text_cleaning_pyspark_agent import TextCleaningPysparkAgent

    # TF-IDF
    from agents.tfidf.tfidf_agent import TFIDFAgent
    from agents.tfidf.tfidf_pyspark_agent import TfidfPsparkAgent

    # Visualizer
    from agents.visualizer.visualizer_agent import VisualizerAgent
    from agents.visualizer.visualizer_sparkagent import VisualizerAgentSpark
    from agents.visualizer.comparison_visualizer import ComparisonVisualizer

    # Configs and Utils
    from config import is_big_data, SNAPSHOT_DIR, PROCESSED_DIR
    from utils.file_handler import read_pandas, write_pandas, detect_file_kind
    from utils.spark_session import get_spark, release_spark, get_spark_info
    from utils.logger import get_logger

    


logger = get_logger(__name__)


class PipelineOrchestrator:
	"""Orchestrates the data preprocessing pipeline with Pandas/PySpark switching."""
	
	def __init__(self, input_file: Path):
		self.input_file = input_file
		self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.use_spark = is_big_data(input_file)      # Determine whether to use Spark
		self.file_kind = detect_file_kind(input_file)

		# Initialize Pandas agents
		self.inspector = GeminiInspectorAgent()
		self.inspector_spark = GeminiInspectorAgentSpark()
		self.imputer = MeanMedianImputerAgent()
		self.normalizer = NormalizeAgent()
		self.scaler = ScalerAgent()
		self.deduper = DedupAgent()
		self.text_processor = TextPreprocessingAgent()
		self.visualizer = VisualizerAgent()
		self.visualizer_spark = VisualizerAgentSpark()
		self.comparison_visualizer = ComparisonVisualizer()
		self.summarizer = SummarizerAgent()
		self.summarizer_spark = SummarizerAgentSpark()
		self.drop_constant_agent = DropConstantFeaturesAgent()
		self.fix_infinite_agent = FixInfiniteAgent()
		self.datetime_agent = HandleDatetimeAgent()
		self.skewness_agent = HandleSkewnessAgent()
		self.lemmatize_agent = LemmatizeTextAgent()
		self.remove_stopwords_agent = RemoveStopwordsAgent()
		self.tfidf_agent = TFIDFAgent()
		self.encoding_agent = EncodingAgent()

		# PySpark agents (lazy init)
		self.imputer_spark = None
		self.normalizer_spark = None
		self.scaler_spark = None
		self.deduper_spark = None
		self.text_processor_spark = None
		self.drop_constant_agent_spark = None
		self.fix_infinite_agent_spark = None
		self.datetime_agent_spark = None
		self.skewness_agent_spark = None
		self.lemmatize_agent_spark = None
		self.remove_stopwords_agent_spark = None
		self.tfidf_agent_spark = None
		self.encoding_agent_spark = None

		# Pipeline state
		self.current_df = None
		self.original_df = None  # Store original dataframe for comparison (Pandas or Pandas sample from Spark)
		self.original_spark_df = None  # Store original Spark DataFrame (if using Spark)
		self.original_column_types = {}  # Track original column types (categorical vs numerical)
		self.label_encoded_columns = set()  # Track columns that were label-encoded (should not be scaled)
		self.onehot_encoded_columns = set()  # Track columns that were one-hot encoded (binary features)
		self.inspection_results = None
		self.suggestions = None
		self.snapshots = []
		self._df_cached = False
		self.spark = None
		self._spark_session_managed = False
		# Track actually applied operations for accurate reporting
		self.applied_ops = []
		# Track generated artifact filenames
		self.report_filename = None
  
		logger.info(f"Initialized orchestrator for {input_file.name} (Spark: {self.use_spark})")


	
	async def run_pipeline(self) -> AsyncGenerator[Dict[str, Any], None]:
		"""Run the complete preprocessing pipeline with progress streaming."""
		try:
			# Step 1: Load data
			yield {"type": "progress", "message": "Loading dataset...", "progress": 10}
			await self._load_data()
			logger.info(f"Dataset loaded. Spark enabled: {self.use_spark}")
			
			# Save original dataframe for manual comparison
			original_path = PROCESSED_DIR / f"original_{self.session_id}.csv"
			self.original_df.to_csv(original_path, index=False)
			logger.info(f"Saved original dataset to: {original_path}")

			# Step 2: Inspect data
			yield {"type": "progress", "message": "Analyzing data with AI...", "progress": 20}
			await self._inspect_data()
	
			# Step 3: Apply preprocessing agents
			yield {"type": "progress", "message": "Applying preprocessing steps...", "progress": 30}
			await self._apply_preprocessing_agents(use_spark=self.use_spark)
	
			# Step 4: Generate visualizations
			yield {"type": "progress", "message": "Generating visualizations...", "progress": 80}
			await self._generate_visualizations()
	
			# Step 5: Create summary report
			yield {"type": "progress", "message": "Creating summary report...", "progress": 85}
			await self._create_summary_report()
		
			# Step 5.5: Generate before/after comparison report
			yield {"type": "progress", "message": "Generating before/after comparison...", "progress": 90}
			await self._generate_comparison_report()

			# Step 6: Save final dataset
			yield {"type": "progress", "message": "Saving processed dataset...", "progress": 95}
			final_path = await self._save_final_dataset()
	
			logger.info(f"Preparing completion message with output_file: {final_path.name}")

			# Complete - include all generated files for download
			completion_data = {
				"type": "complete",
				"message": "Pipeline completed successfully!",
				"progress": 100,
				"output_file": final_path.name,
				"report_file": self.report_filename or (f"report_{self.session_id}.html" if not self.use_spark else f"report_spark_{self.session_id}.html"),
				"visualization_gallery": f"visualizations_gallery_{self.session_id}.html",
				"comparison_report": f"comparison_report_{self.session_id}.html"
			}
			logger.info(f"Sending completion message: {completion_data}")
			yield completion_data
			logger.info("Completion message yielded, generator ending")
			return  # Explicitly end the generator here
	
		except Exception as e:
			logger.error(f"Pipeline failed: {e}", exc_info=True)
			yield {
				"type": "error",
				"message": f"Pipeline failed: {str(e)}",
				"progress": 0
			}
			return  # Explicitly end the generator here
   
		finally:
			# Cleanup Spark resources
			if self.use_spark:
				if self._df_cached and hasattr(self, "current_df") and self.current_df is not None:
					try:
						self.current_df.unpersist()
						logger.info("Spark DataFrame unpersisted successfully.")
					except Exception as e:
						logger.warning(f"Failed to unpersist DataFrame: {e}")
				
				# Release Spark session reference
				if self._spark_session_managed:
					try:
						release_spark()
						logger.info("Spark session reference released.")
					except Exception as e:
						logger.warning(f"Failed to release Spark session: {e}")
				
				# Log session info for debugging
				session_info = get_spark_info()
				logger.info(f"Spark session info: {session_info}")
	
	async def _load_data(self):
		"""Load the input dataset."""
		if self.use_spark:
			await self._load_with_spark()
		else:
			await self._load_with_pandas()
		
		# Store original dataframe and track column types for visualization
		if self.use_spark:
			row_count = self.current_df.count()
			col_count = len(self.current_df.columns)
			logger.info(f"Loaded Spark dataset with shape: ({row_count}, {col_count})")
			
			# Store original Spark DataFrame for visualization
			self.original_spark_df = self.current_df
			
			# For comparison report, take a sample and convert to Pandas
			sample_size = min(10000, row_count)  # Sample up to 10k rows
			self.original_df = self.current_df.limit(sample_size).toPandas()
			logger.info(f"Sampled {len(self.original_df)} rows from Spark DataFrame for comparison report")
			
			# Track original column types from Spark schema (more reliable than sample)
			for field in self.current_df.schema.fields:
				col_name = field.name
				# Check if numeric type in Spark using isinstance
				if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType, 
											   DecimalType, ShortType, ByteType)):
					self.original_column_types[col_name] = 'numerical'
				else:
					self.original_column_types[col_name] = 'categorical'
		else:
			logger.info(f"Loaded Pandas dataset with shape: {self.current_df.shape}")
			# Make a deep copy to ensure original is never modified
			self.original_df = self.current_df.copy(deep=True)
			
			# Track original column types (categorical vs numerical)
			for col in self.original_df.columns:
				if pd.api.types.is_numeric_dtype(self.original_df[col]):
					self.original_column_types[col] = 'numerical'
				else:
					self.original_column_types[col] = 'categorical'
		
		logger.info(f"Tracked original column types: {sum(1 for t in self.original_column_types.values() if t == 'numerical')} numerical, {sum(1 for t in self.original_column_types.values() if t == 'categorical')} categorical")
	
	async def _load_with_pandas(self):
		"""Load data using Pandas."""
		self.current_df = read_pandas(self.input_file, self.file_kind)
	
	async def _load_with_spark(self):
		"""Load data using PySpark with proper session management."""
		try:
			# Get Spark session with proper lifecycle management
			self.spark = get_spark(app_name=f"AIDataCleaner_{self.session_id}")
			self._spark_session_managed = True
			logger.info(f"Acquired Spark session for pipeline {self.session_id}")

			# Load data based on file type
			if self.file_kind == "csv":
				self.current_df = self.spark.read.option("header", "true") \
												.option("inferSchema", "true") \
												.csv(str(self.input_file))
			elif self.file_kind == "json":
				self.current_df = self.spark.read.json(str(self.input_file))
			else:  # parquet
				self.current_df = self.spark.read.parquet(str(self.input_file))

			# Cache the DataFrame to speed up repeated actions
			self.current_df = self.current_df.cache()
			self._df_cached = True

			# Force materialization of cache with count() action
			row_count = self.current_df.count()
			col_count = len(self.current_df.columns)
			logger.info(f"[Spark] Loaded and cached dataset with {row_count} rows and {col_count} columns")

			# Initialize PySpark agents now that Spark session is ready
			self._initialize_spark_agents()

		except Exception as e:
			logger.error(f"Failed to load Spark dataset: {e}")
			# Release session reference on error
			if self._spark_session_managed:
				release_spark()
				self._spark_session_managed = False
			raise
	def _initialize_spark_agents(self):
		"""Initialize PySpark agents with the Spark session."""
		logger.info("Initializing PySpark agents...")
		self.imputer_spark = ImputersAgent(spark=self.spark, df=None)
		self.normalizer_spark = NormalizationAgentPyspark(spark=self.spark, df=None)
		self.scaler_spark = ScalingAgentPyspark(spark=self.spark, df=None)
		self.deduper_spark = DedupAgentSpark(spark=self.spark, df=None)
		self.text_processor_spark = TextCleaningPysparkAgent(spark=self.spark, df=None)
		self.drop_constant_agent_spark = DropConstantFeaturesAgentPyspark(spark=self.spark, df=None)
		self.fix_infinite_agent_spark = FixInfiniteValuesAgentPyspark(spark=self.spark, df=None)
		self.datetime_agent_spark = HandleDatetimeAgentPyspark(spark=self.spark, df=None)
		self.skewness_agent_spark = HandleSkewnessAgentPyspark(spark=self.spark, df=None)
		self.lemmatize_agent_spark = LemmatizationAgentPyspark(spark=self.spark, df=None)
		self.remove_stopwords_agent_spark = RemoveStopwordsAgentPyspark(spark=self.spark, df=None)
		self.tfidf_agent_spark = TfidfPsparkAgent(spark=self.spark, df=None)
		self.encoding_agent_spark = EncodingAgentPyspark(spark=self.spark, df=None)
		logger.info("PySpark agents initialized successfully.")

	
	async def _inspect_data(self):
		"""Run data inspection and get preprocessing suggestions."""
		try:
			if self.use_spark:
				logger.info("Using PySpark Inspector Agent for inspection...")
				self.inspection_results = self.inspector_spark.process(self.current_df)
			else:
				logger.info("Using Pandas Inspector Agent for inspection...")
				self.inspection_results = self.inspector.process(self.current_df)
			
			# Normalize suggestions to a dict shape: { column: [ {suggestion, ...}, ... ] }
			self.suggestions = self.inspection_results.get("suggestions", {}) if isinstance(self.inspection_results, dict) else {}
			if not isinstance(self.suggestions, dict):
				logger.warning(f"Inspector returned non-dict suggestions ({type(self.suggestions).__name__}). Normalizing to empty dict.")
				self.suggestions = {}
			logger.info(f"Inspection completed with {len(self.suggestions)} column suggestions.")
	
		except Exception as e:
			logger.error(f"Data inspection failed: {e}")
			self.inspection_results, self.suggestions = {}, {}

	async def _apply_preprocessing_agents(self, use_spark: bool):
		"""Apply preprocessing agents based on suggestions, supporting both Pandas and PySpark."""
		# Guard: ensure suggestions is a dict
		if not isinstance(self.suggestions, dict):
			logger.warning(f"Suggestions is not a dict ({type(self.suggestions).__name__}); skipping preprocessing.")
			return

		for col, actions in self.suggestions.items():
			# Ensure actions is a list (support multi-action per column)
			if not isinstance(actions, list):
				actions = [actions]

			# Enforce the exact operation sequence requested
			order_index = {
				"fix_infinite": 10,
				"remove_outliers": 20,
				"fill_missing": 30,
				"deduplicate": 40,
				"drop_constant": 50,
				"handle_datetime": 60,
				"handle_skewness": 70,
				"categorical_encode": 80,
				"text_clean": 90,
				"remove_stopwords": 100,
				"lemmatize": 110,
				"tfidf": 120,
				"standardize": 130,
				# "normalize_range": 130,
				"skip": 0,
			}

			try:
				actions = sorted(actions, key=lambda a: order_index.get(a.get("suggestion"), 1000))
			except Exception:
				pass

			for action in actions:
				if action["suggestion"] == "skip":
					logger.info(f"Skipping column '{col}' as per suggestion.")
					continue

				# Skip scaling/standardization for encoded categorical columns
				if action["suggestion"] in ["standardize", "normalize_range"]:
					if col in self.label_encoded_columns:
						logger.info(f"Skipping {action['suggestion']} for label-encoded column '{col}' (categorical feature)")
						continue
					if col in self.onehot_encoded_columns:
						logger.info(f"Skipping {action['suggestion']} for one-hot encoded column '{col}' (binary feature)")
						continue
				
				# Skip skewness handling for encoded categorical columns
				if action["suggestion"] == "handle_skewness":
					# Check if it's a label-encoded column
					if col in self.label_encoded_columns:
						logger.info(f"Skipping skewness handling for label-encoded column '{col}' (categorical feature)")
						continue
					
					# Check if it's a one-hot encoded column
					if col in self.onehot_encoded_columns:
						logger.info(f"Skipping skewness handling for one-hot encoded column '{col}' (binary feature)")
						continue
					
					# Check if it's a binary column (0/1 values)
					if not self.use_spark:
						if col in self.current_df.columns:
							unique_vals = self.current_df[col].dropna().unique()
							if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
								logger.info(f"Skipping skewness handling for binary column '{col}'")
								continue
					else:
						# For Spark, check distinct count
						distinct_count = self.current_df.select(col).distinct().count()
						if distinct_count <= 2:
							logger.info(f"Skipping skewness handling for binary column '{col}' (Spark)")
							continue

				# Select agent based on Spark usage
				agent = self._get_agent_for_suggestion(action["suggestion"], col, use_spark=self.use_spark)
				if agent is None:
					logger.warning(f"No agent found for suggestion '{action['suggestion']}' on column '{col}'")
					continue

				try:
					# Validate data before processing
					before_shape = self.current_df.shape if not self.use_spark else (self.current_df.count(), len(self.current_df.columns))

					# Prepare parameters with proper method setting
					params = action.get("parameters", {})

					# Set method parameter for agents based on suggestion
					if action["suggestion"] == "remove_outliers":
						params["method"] = "remove_outliers"
						params.setdefault("outlier_strategy", "median")  # options: median | clip | nan
					elif action["suggestion"] == "standardize":
						params["method"] = "standard"
					elif action["suggestion"] == "normalize_range":
						params["method"] = "minmax"
					elif action["suggestion"] == "categorical_encode":
						# Save Gemini's preference before overwriting
						gemini_method = params.get("method", None)
						params["method"] = "categorical_encode"
						
						# Map Gemini's method parameter to strategy
						if gemini_method in ["label_encode", "label"]:
							params["strategy"] = "label"
							logger.info(f"Column '{col}': Using LABEL encoding (Gemini recommended for high cardinality)")
						elif gemini_method in ["onehot_encode", "onehot"]:
							params["strategy"] = "onehot"
							logger.info(f"Column '{col}': Using ONE-HOT encoding (Gemini recommended)")
						else:
							params.setdefault("strategy", "onehot")  # Default to onehot encoding
							logger.info(f"Column '{col}': Using ONE-HOT encoding (default)")
						params.setdefault("keep_original", False)

					# Guard: ensure the target column exists for column-bound operations
					column_bound_ops = {
						"fill_missing",
						"fix_infinite",
						"remove_outliers",
						"drop_constant",
						"handle_datetime",
						"handle_skewness",
						"normalize_range",
						"standardize",
						"categorical_encode",
						"text_clean",
						"remove_stopwords",
						"lemmatize",
						"tfidf",
					}
					if action["suggestion"] in column_bound_ops:
						if self.use_spark:
							if col not in self.current_df.columns:
								logger.warning(f"Column '{col}' missing before '{action['suggestion']}', skipping action.")
								continue
						else:
							if col not in self.current_df.columns:
								logger.warning(f"Column '{col}' missing before '{action['suggestion']}', skipping action.")
								continue

					# --------------------------------------------------------
					# ✅ Proper indentation begins here (fixed)
					# --------------------------------------------------------
					if self.use_spark:
						# PySpark processing
						logger.info(f"[Spark] Processing column '{col}' with {agent.__class__.__name__}")
						# Special handling for categorical encoding to track metadata
						if action["suggestion"] == "categorical_encode":
							params["return_metadata"] = True
							result = agent.process(
								self.current_df,
								column=col,
								**params
							)
							if isinstance(result, tuple):
								self.current_df, encoding_metadata = result
								self._update_column_types_after_encoding(encoding_metadata)
							else:
								self.current_df = result
								logger.warning("Encoding agent did not return metadata")
						else:
							self.current_df = agent.process(
								self.current_df,
								column=col,
								**params
							)

						# Persist transformed DF
						try:
							self.current_df = self.current_df.persist()
							self._df_cached = True
							logger.debug("Persisted Spark DataFrame after transformation")
						except Exception as e:
							logger.warning(f"Failed to persist DataFrame after {agent.__class__.__name__}: {e}")

					else:
						# Pandas processing
						logger.info(f"[Pandas] Processing column '{col}' with {agent.__class__.__name__}")
						# Special handling for categorical encoding to track metadata
						if action["suggestion"] == "categorical_encode":
							params["return_metadata"] = True
							result = agent.process(
								self.current_df,
								column=col,
								**params
							)
							if isinstance(result, tuple):
								self.current_df, encoding_metadata = result
								self._update_column_types_after_encoding(encoding_metadata)
							else:
								self.current_df = result
								logger.warning("Encoding agent did not return metadata")
						else:
							self.current_df = agent.process(
								self.current_df,
								column=col,
								**params
							)

					# Validate data after processing
					after_shape = self.current_df.shape if not self.use_spark else (self.current_df.count(), len(self.current_df.columns))

					# Check for data corruption (concatenated strings)
					if not self.use_spark and col in self.current_df.columns:
						if self.current_df[col].dtype == "object":
							max_length = self.current_df[col].astype(str).str.len().max()
							if max_length > 1000:  # Likely concatenated data
								logger.error(f"Column '{col}' contains concatenated data after {action['suggestion']} (max length: {max_length}). Reverting operation.")
								continue

					# Track applied operation
					self.applied_ops.append({
						"column": col, 
						"operation": action["suggestion"],
						"reason": action.get("reason", "")
					})
					logger.info(f"Applied {agent.__class__.__name__} to column '{col}' (shape: {before_shape} -> {after_shape})")

					# Optionally save snapshot
					# await self._save_snapshot(f"{col}_{action['suggestion']}_{datetime.now().strftime('%H%M%S')}")

				except Exception as e:
					logger.error(f"Failed to process column '{col}' with {agent.__class__.__name__}: {e}")


	 
	def _update_column_types_after_encoding(self, encoding_metadata: dict):
		"""Update original_column_types after categorical encoding.
		
		Args:
			encoding_metadata: Dictionary containing encoding information
				{'method': 'onehot'/'label', 'encoded_columns': {col: {'new_columns': [...], 'removed': bool}}}
		"""
		logger.info("Updating column types after categorical encoding...")
		encoding_method = encoding_metadata.get('method', 'unknown')
		
		for original_col, info in encoding_metadata['encoded_columns'].items():
			new_columns = info['new_columns']
			was_removed = info['removed']
			
			# Track label-encoded columns (should not be scaled/skewed)
			if encoding_method == 'label':
				for new_col in new_columns:
					self.label_encoded_columns.add(new_col)
					logger.info(f"Marked '{new_col}' as label-encoded (will skip scaling/skewness)")
			
			# Track one-hot encoded columns (should not be scaled/skewed)
			elif encoding_method == 'onehot':
				for new_col in new_columns:
					self.onehot_encoded_columns.add(new_col)
					logger.info(f"Marked '{new_col}' as one-hot encoded (will skip scaling/skewness)")
			
			# If original column was removed, delete it from tracking
			if was_removed and original_col in self.original_column_types:
				logger.info(f"Removing '{original_col}' from column type tracking (replaced by encoding)")
				del self.original_column_types[original_col]
			
			# Add all new encoded columns as numerical
			for new_col in new_columns:
				if new_col not in self.original_column_types:
					self.original_column_types[new_col] = 'numerical'
					logger.info(f"Added encoded column '{new_col}' as numerical type")
				else:
					# Column exists but type changed (label encoding case)
					if self.original_column_types[new_col] == 'categorical':
						self.original_column_types[new_col] = 'numerical'
						logger.info(f"Updated column '{new_col}' from categorical to numerical (label encoded)")
		
		logger.info(f"Column types updated. Current tracking: {sum(1 for t in self.original_column_types.values() if t == 'numerical')} numerical, {sum(1 for t in self.original_column_types.values() if t == 'categorical')} categorical")
	
	def _get_agent_for_suggestion(self, suggestion: str, column: str = None, use_spark: bool = False):
		"""Return the appropriate agent based on suggestion and Spark usage."""
		# Map suggestions to Pandas and Spark agents
		agent_map = {
			"fill_missing": (self.imputer, self.imputer_spark),
			"normalize_range": (self.normalizer, self.normalizer_spark),
			"standardize": (self.scaler, self.scaler_spark),
			"remove_outliers": (self.scaler, self.scaler_spark),  # Could be separate agent
			"deduplicate": (self.deduper, self.deduper_spark),
			"text_clean": (self.text_processor, self.text_processor_spark),
			"drop_constant": (self.drop_constant_agent, self.drop_constant_agent_spark),
			"fix_infinite": (self.fix_infinite_agent, self.fix_infinite_agent_spark),
			"handle_datetime": (self.datetime_agent, self.datetime_agent_spark),
			"handle_skewness": (self.skewness_agent, self.skewness_agent_spark),
			"lemmatize": (self.lemmatize_agent, self.lemmatize_agent_spark),
			"remove_stopwords": (self.remove_stopwords_agent, self.remove_stopwords_agent_spark),
			"tfidf": (self.tfidf_agent, self.tfidf_agent_spark),
            "categorical_encode": (self.encoding_agent, self.encoding_agent_spark),
		}
	
		pandas_agent, spark_agent = agent_map.get(suggestion, (None, None))
		return spark_agent if use_spark else pandas_agent
	
	
	async def _generate_visualizations(self):
		"""Generate data visualizations from ORIGINAL data."""
		try:
			if self.use_spark:
				# Use Spark-native visualizer on ORIGINAL Spark DataFrame
				logger.info("Generating visualizations from ORIGINAL Spark data...")
				visualizations = self.visualizer_spark.process(self.original_spark_df)
			else:
				# Generate visualizations from ORIGINAL Pandas DataFrame
				logger.info("Generating visualizations from ORIGINAL Pandas data...")
				clean_df = self._fix_concatenated_data(self.original_df.copy())
				visualizations = self.visualizer.process(clean_df)
			
			await self._save_visualizations(visualizations)
		except Exception as e:
			logger.error(f"Visualization generation failed: {e}")
			# Return empty visualizations to continue pipeline
			await self._save_visualizations({})
	
	async def _create_summary_report(self):
		"""Create a summary report."""
		try:
			if self.use_spark:
				logger.info("Generating Spark summary report without converting to Pandas...")
				report_data = self.summarizer_spark.process(
					self.current_df,
					inspection_results=self.inspection_results,
					suggestions=self.suggestions,
					snapshots=self.snapshots,
					applied_operations=self.applied_ops
				)
			else:
				report_data = self.summarizer.process(
					self.current_df,
					inspection_results=self.inspection_results,
					suggestions=self.suggestions,
					snapshots=self.snapshots,
					applied_operations=self.applied_ops
				)
			await self._save_report(report_data)
		except Exception as e:
			logger.error(f"Report generation failed: {e}")

	async def _generate_comparison_report(self):
		"""Generate before/after comparison visualizations and report."""
		try:
			logger.info("Generating before/after comparison report...")
			
			# Convert Spark to Pandas if needed (with sampling to avoid memory overflow)
			if self.use_spark:
				row_count = self.current_df.count()
				sample_size = min(10000, row_count)
				logger.info(f"Sampling {sample_size} rows from Spark DataFrame for comparison (avoiding memory overflow)...")
				processed_pandas = self.current_df.limit(sample_size).toPandas()
				
				# Also ensure original_df is same size (it should already be sampled from _load_data)
				if len(self.original_df) > sample_size:
					logger.info(f"Trimming original_df to {sample_size} rows for fair comparison")
					self.original_df = self.original_df.head(sample_size)
			else:
				processed_pandas = self.current_df
		
			# Generate comparison
			comparison_result = self.comparison_visualizer.process(
				original_df=self.original_df,
				processed_df=processed_pandas,
				original_column_types=self.original_column_types,
				session_id=self.session_id
			)
		
			# Save comparison report
			comparison_path = PROCESSED_DIR / f"comparison_report_{self.session_id}.html"
			comparison_path.write_text(comparison_result["report_html"], encoding='utf-8')
			logger.info(f"Saved comparison report to: {comparison_path}")
		
			# Add to snapshots
			self.snapshots.append({
				"step": "comparison_report",
				"path": str(comparison_path),
				"columns_compared": len(comparison_result["visualizations"]),
				"timestamp": datetime.now().isoformat()
			})
		
		except Exception as e:
			logger.error(f"Comparison report generation failed: {e}", exc_info=True)
   
#completed pandas+pyspark
	async def _save_final_dataset(self) -> Path:
		"""Save the final processed dataset using native Spark or Pandas writers."""
		output_filename = f"processed_{self.session_id}"
		
		if self.use_spark:
			# Use native Spark writers (winutils.exe is installed)
			row_count = self.current_df.count()
			col_count = len(self.current_df.columns)
			logger.info(f"[Spark] Saving dataset with {row_count} rows and {col_count} columns using native Spark writer...")
			
			# Convert vector columns to regular columns for CSV/JSON compatibility
			df_to_save = self._convert_vectors_to_columns(self.current_df)
			
			# Prepare output path
			output_path = PROCESSED_DIR / output_filename
			
			# Use native Spark writers based on file format
			if self.file_kind == "csv":
				# Coalesce to single file for easier handling
				df_to_save.coalesce(1).write.mode("overwrite") \
					.option("header", "true") \
					.csv(str(output_path))
				# Find the actual CSV file in the output directory
				csv_files = list(output_path.glob("*.csv"))
				if csv_files:
					final_path = PROCESSED_DIR / f"{output_filename}.csv"
					csv_files[0].rename(final_path)
					# Clean up the directory
					shutil.rmtree(output_path, ignore_errors=True)
					output_path = final_path
				
			elif self.file_kind == "parquet":
				df_to_save.write.mode("overwrite").parquet(str(output_path))
				
			elif self.file_kind == "json":
				df_to_save.coalesce(1).write.mode("overwrite").json(str(output_path))
				# Find the actual JSON file in the output directory
				json_files = list(output_path.glob("*.json"))
				if json_files:
					final_path = PROCESSED_DIR / f"{output_filename}.json"
					json_files[0].rename(final_path)
					# Clean up the directory
					shutil.rmtree(output_path, ignore_errors=True)
					output_path = final_path
			
			logger.info(f"[Spark] Saved final dataset to '{output_path}'")
			return output_path
		else:
			# Save Pandas DataFrame in original format
			path = write_pandas(self.current_df, f"{output_filename}.{self.file_kind}", self.file_kind)
			logger.info(f"[Pandas] Saved final dataset '{path}' with shape {self.current_df.shape}")
			return Path(path) if not isinstance(path, Path) else path
	
	def _convert_vectors_to_columns(self, df):
		"""Convert SparseVector/DenseVector columns to regular columns for CSV compatibility."""
		from pyspark.ml.linalg import VectorUDT
		from pyspark.ml.functions import vector_to_array
		from pyspark.sql import functions as F
		
		# Check each column for vector types
		vector_cols = []
		for field in df.schema.fields:
			if isinstance(field.dataType, VectorUDT):
				vector_cols.append(field.name)
		
		if not vector_cols:
			return df
		
		logger.info(f"Converting {len(vector_cols)} vector column(s) to individual columns: {vector_cols}")
		
		for col_name in vector_cols:
			# Convert vector to array first
			df = df.withColumn(f"{col_name}_array", vector_to_array(F.col(col_name)))
			
			# Get the vector size by examining the first non-null value
			first_row = df.select(f"{col_name}_array").filter(F.col(f"{col_name}_array").isNotNull()).first()
			if first_row and first_row[0] is not None:
				vector_size = len(first_row[0])
				
				# Create individual columns for each array element
				for i in range(vector_size):
					new_col_name = f"{col_name}_{i}"
					df = df.withColumn(new_col_name, F.col(f"{col_name}_array")[i])
				
				# Drop the temporary array and original vector columns
				df = df.drop(col_name, f"{col_name}_array")
				logger.info(f"Expanded '{col_name}' into {vector_size} columns")
		
		return df
	
#completed pandas+pyspark 
	async def _save_snapshot(self, step_name: str):
		"""Save snapshot of current dataframe in Spark or Pandas (optimized)."""
		snapshot_filename = f"snapshot_{self.session_id}_{step_name}.{self.file_kind}"
		
		if self.use_spark:
			# Use checkpoint instead of write for intermediate snapshots (more efficient)
			path = str(PROCESSED_DIR / snapshot_filename)
			# Note: count() reuses cache, no full scan needed
			row_count = self.current_df.count()
			col_count = len(self.current_df.columns)
			
			# Only write to disk for critical snapshots (initial_load, final)
			if "initial" in step_name or "final" in step_name:
				self.current_df.write.mode("overwrite").parquet(path)
				logger.info(f"[Spark] Saved snapshot '{step_name}' to disk with shape ({row_count}, {col_count})")
			else:
				logger.info(f"[Spark] Skipped disk write for intermediate snapshot '{step_name}' (shape: {row_count}, {col_count})")
			
			self.snapshots.append({
				"step": step_name,
				"path": path if "initial" in step_name or "final" in step_name else None,
				"shape": (row_count, col_count),
				"timestamp": datetime.now().isoformat()
			})
		else:
			path = write_pandas(self.current_df, snapshot_filename, self.file_kind)
			self.snapshots.append({
				"step": step_name,
				"path": path,
				"shape": self.current_df.shape,
				"timestamp": datetime.now().isoformat()
			})
			logger.info(f"[Pandas] Saved snapshot '{step_name}'")
	
	async def _save_visualizations(self, visualizations: Dict[str, Any]):
		"""Save generated visualizations and create gallery HTML."""
		try:
			# Visualizer returns dict of {name: data_uri}
			viz_dir = PROCESSED_DIR / "visualizations"
			viz_dir.mkdir(exist_ok=True)
			saved = []
			for name, data_uri in (visualizations or {}).items():
				if isinstance(data_uri, str) and data_uri.startswith("data:image"):
					# Save base64 image
					header, b64data = data_uri.split(',', 1)
					img_path = viz_dir / f"{self.session_id}_{name}.png"
					with open(img_path, 'wb') as f:
						import base64
						f.write(base64.b64decode(b64data))
					saved.append(str(img_path))
				else:
					# If it's a filepath or other content, store reference only
					saved.append(str(data_uri))
			
			# Generate visualization gallery HTML
			gallery_html = self._generate_visualization_gallery(visualizations)
			gallery_path = PROCESSED_DIR / f"visualizations_gallery_{self.session_id}.html"
			gallery_path.write_text(gallery_html, encoding='utf-8')
			
			self.snapshots.append({
				"step": "visualizations",
				"path": str(viz_dir),
				"files": saved,
				"gallery_path": str(gallery_path),
				"timestamp": datetime.now().isoformat()
			})
			logger.info(f"Saved {len(saved)} visualization files to {viz_dir}")
			logger.info(f"Generated visualization gallery: {gallery_path}")
		except Exception as e:
			logger.error(f"Failed to save visualizations: {e}")
	
	def _generate_visualization_gallery(self, visualizations: Dict[str, Any]) -> str:
		"""Generate HTML gallery with before/after toggle buttons."""
		# Group visualizations by type and column
		grouped_viz = {}
		for name, data_uri in visualizations.items():
			# Parse visualization name to extract column and type
			if '_before' in name:
				base_name = name.replace('_before', '')
				viz_type = 'before'
			elif '_after' in name:
				base_name = name.replace('_after', '')
				viz_type = 'after'
			else:
				base_name = name
				viz_type = 'single'
			
			if base_name not in grouped_viz:
				grouped_viz[base_name] = {}
			grouped_viz[base_name][viz_type] = data_uri
		
		html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization Gallery - {self.session_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 40px; 
            border-radius: 12px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 4px solid #667eea; 
            padding-bottom: 15px; 
            margin-bottom: 30px; 
            font-size: 2.5em; 
            text-align: center;
        }}
        .gallery-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); 
            gap: 30px; 
            margin: 30px 0; 
        }}
        .viz-card {{ 
            background: white; 
            border: 2px solid #e0e0e0; 
            border-radius: 12px; 
            padding: 25px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .viz-card:hover {{ 
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        .viz-title {{ 
            font-size: 1.4em; 
            font-weight: bold; 
            color: #2c3e50; 
            margin-bottom: 20px; 
            text-align: center;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .toggle-container {{ 
            display: flex; 
            justify-content: center; 
            margin-bottom: 20px; 
            gap: 10px;
        }}
        .toggle-btn {{ 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: all 0.3s ease;
            font-size: 1em;
        }}
        .toggle-btn.before {{ 
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
            color: white; 
        }}
        .toggle-btn.after {{ 
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%); 
            color: white; 
        }}
        .toggle-btn.single {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
        }}
        .toggle-btn.active {{ 
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .toggle-btn:hover {{ 
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .viz-image {{ 
            width: 100%; 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: none;
        }}
        .viz-image.active {{ 
            display: block;
        }}
        .comparison-info {{ 
            text-align: center; 
            margin-top: 15px; 
            padding: 10px; 
            background: #f8f9fa; 
            border-radius: 6px; 
            font-size: 0.9em; 
            color: #666;
        }}
        .navbar {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 15px 40px; 
            margin: -40px -40px 30px -40px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }}
        .navbar-title {{ 
            color: white; 
            font-size: 1.2em; 
            font-weight: 600; 
        }}
        .navbar-links {{ 
            display: flex; 
            gap: 15px; 
        }}
        .nav-btn {{ 
            padding: 10px 20px; 
            background: rgba(255,255,255,0.2); 
            color: white; 
            text-decoration: none; 
            border-radius: 25px; 
            transition: all 0.3s; 
            font-weight: 600; 
            border: 2px solid rgba(255,255,255,0.3); 
        }}
        .nav-btn:hover {{ 
            background: white; 
            color: #667eea; 
            transform: translateY(-2px); 
            box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
        }}
        .nav-btn.active {{ 
            background: white; 
            color: #667eea; 
        }}
        @media print {{ 
            body {{ background: white; }} 
            .container {{ box-shadow: none; }} 
            .navbar {{ display: none; }} 
            .toggle-container {{ display: none; }}
            .viz-image {{ display: block !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="navbar">
            <div class="navbar-title">📊 Visualization Gallery</div>
            <div class="navbar-links">
                <a href="report_{self.session_id}.html" class="nav-btn">📄 Report</a>
                <a href="#" class="nav-btn active">📊 Gallery</a>
                <a href="#" onclick="window.print(); return false;" class="nav-btn">🖨️ Print</a>
            </div>
        </div>
        
        <h1>📊 Data Visualization Gallery</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Interactive before/after comparisons of your data cleaning process
        </p>
        
        <div class="gallery-grid">"""
		
		# Generate visualization cards
		for base_name, viz_data in grouped_viz.items():
			# Clean up the base name for display
			display_name = base_name.replace('_', ' ').title()
			
			html += f"""
            <div class="viz-card">
                <div class="viz-title">{display_name}</div>
                <div class="toggle-container">"""
			
			# Add toggle buttons based on available visualizations
			if 'before' in viz_data and 'after' in viz_data:
				html += f"""
                    <button class="toggle-btn before active" onclick="toggleViz('{base_name}', 'before')">🔴 Before</button>
                    <button class="toggle-btn after" onclick="toggleViz('{base_name}', 'after')">🟢 After</button>"""
				html += f"""
                </div>
                <img id="{base_name}_before" class="viz-image active" src="{viz_data['before']}" alt="{display_name} - Before">
                <img id="{base_name}_after" class="viz-image" src="{viz_data['after']}" alt="{display_name} - After">
                <div class="comparison-info">
                    Click buttons above to compare before and after data cleaning
                </div>"""
			elif 'before' in viz_data:
				html += f"""
                    <button class="toggle-btn before active" onclick="toggleViz('{base_name}', 'before')">🔴 Before</button>"""
				html += f"""
                </div>
                <img id="{base_name}_before" class="viz-image active" src="{viz_data['before']}" alt="{display_name} - Before">
                <div class="comparison-info">
                    Before data cleaning visualization
                </div>"""
			elif 'after' in viz_data:
				html += f"""
                    <button class="toggle-btn after active" onclick="toggleViz('{base_name}', 'after')">🟢 After</button>"""
				html += f"""
                </div>
                <img id="{base_name}_after" class="viz-image active" src="{viz_data['after']}" alt="{display_name} - After">
                <div class="comparison-info">
                    After data cleaning visualization
                </div>"""
			else:
				# Single visualization
				viz_type = list(viz_data.keys())[0]
				html += f"""
                    <button class="toggle-btn single active" onclick="toggleViz('{base_name}', '{viz_type}')">📊 View</button>"""
				html += f"""
                </div>
                <img id="{base_name}_{viz_type}" class="viz-image active" src="{viz_data[viz_type]}" alt="{display_name}">
                <div class="comparison-info">
                    Data visualization
                </div>"""
			
			html += """
            </div>"""
		
		html += """
        </div>
        
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #666;">
            Visualization Gallery generated by AI Data Cleaner | 
            <a href="#" onclick="window.print()">Print Gallery</a>
        </p>
    </div>
    
    <script>
        function toggleViz(baseName, type) {
            // Hide all images for this visualization
            const beforeImg = document.getElementById(baseName + '_before');
            const afterImg = document.getElementById(baseName + '_after');
            const singleImg = document.getElementById(baseName + '_' + type);
            
            if (beforeImg) beforeImg.classList.remove('active');
            if (afterImg) afterImg.classList.remove('active');
            if (singleImg) singleImg.classList.remove('active');
            
            // Show the selected image
            const targetImg = document.getElementById(baseName + '_' + type);
            if (targetImg) {
                targetImg.classList.add('active');
            }
            
            // Update button states
            const buttons = document.querySelectorAll(`[onclick*="${baseName}"]`);
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        // Initialize all visualizations to show the first available option
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.viz-card');
            cards.forEach(card => {
                const firstBtn = card.querySelector('.toggle-btn');
                if (firstBtn) {
                    firstBtn.click();
                }
            });
        });
    </script>
</body>
</html>"""
		
		return html

	def _fix_concatenated_data(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Fix concatenated categorical data by splitting and taking the first value."""
		clean_df = df.copy()
		
		for col in clean_df.select_dtypes(include=['object']).columns:
			try:
				# Check for concatenated data
				max_length = clean_df[col].astype(str).str.len().max()
				if max_length > 100:  # Likely concatenated
					logger.warning(f"Fixing concatenated data in column '{col}' (max length: {max_length})")
					
					# Try to extract the first categorical value from concatenated string
					def extract_first_category(value):
						if pd.isna(value) or value == '':
							return value
						value_str = str(value)
						# Look for common categorical patterns
						for pattern in ['Male', 'Female', 'Other', 'Yes', 'No', 'True', 'False']:
							if pattern in value_str:
								return pattern
						# If no pattern found, take first 20 characters
						return value_str[:20] if len(value_str) > 20 else value_str
					
					clean_df[col] = clean_df[col].apply(extract_first_category)
					logger.info(f"Fixed concatenated data in column '{col}'")
			except Exception as e:
				logger.warning(f"Could not fix column '{col}': {e}. Removing from visualization.")
				clean_df = clean_df.drop(columns=[col])
		
		return clean_df

	async def _save_report(self, report_data):
		"""Save the summary report."""
		try:
			# If it's a list, take the first item
			if isinstance(report_data, list) and report_data:
				report_data = report_data[0]

			# Ensure we have a dict now
			if not isinstance(report_data, dict):
				logger.warning(f"Summarizer returned unexpected type: {type(report_data).__name__}")
				return

			path = report_data.get("report_path")
			html = report_data.get("html_content")

			if path:
				try:
					from pathlib import Path as _Path
					self.report_filename = _Path(path).name
				except Exception:
					self.report_filename = str(path).split('/')[-1]
				logger.info(f"Report saved at: {path} (filename set to {self.report_filename})")
				return

			if html:
				report_filename = (
					f"report_spark_{self.session_id}.html" if self.use_spark else f"report_{self.session_id}.html"
				)
				report_path = PROCESSED_DIR / report_filename
				report_path.write_text(html, encoding='utf-8')
				self.report_filename = report_filename
				logger.info(f"Report saved at: {report_path} (filename set to {self.report_filename})")
				return

			logger.warning("Summarizer did not return a report path or html content")

		except Exception as e:
			logger.error(f"Failed to save report: {e}")
