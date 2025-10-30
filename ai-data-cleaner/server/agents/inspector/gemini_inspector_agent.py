import json
import time
from typing import Dict, Any, List

import google.generativeai as genai
import pandas as pd

try:
    from .inspector_agent import InspectorAgent
    from ...config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
    from ...utils.logger import get_logger
except ImportError:
    from agents.inspector.inspector_agent import InspectorAgent
    from config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
    from utils.logger import get_logger

logger = get_logger(__name__)

class GeminiInspectorAgent(InspectorAgent):
    """Inspector agent that uses Gemini to analyze data samples and suggest preprocessing steps."""

    def __init__(self):
        super().__init__()
        self.name = "GeminiInspectorAgent"
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set, falling back to heuristic inspection")
            self.gemini_available = False
        else:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini_available = True
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def process(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze dataframe and return preprocessing suggestions."""
        inspection_data = super().process(df, **kwargs)

        if not self.gemini_available:
            return self._heuristic_analysis(inspection_data)

        try:
            prompt = self._create_analysis_prompt(inspection_data)
            
            
            # Log the prompt sent to Gemini for traceability (debug level to avoid noise)
            logger.debug("*************** GEMINI PROMPT (PANDAS) ***************\n%s\n***************************************************", prompt)
            
            response = self._safe_generate_content(prompt)
            # Log raw Gemini response text so it is visible in console
            try:
                logger.info("GEMINI RAW RESPONSE (PANDAS): %s", getattr(response, "text", str(response)))
            except Exception as log_exc:
                logger.warning(f"Failed to log raw Gemini response: {log_exc}")
            suggestions = self._parse_gemini_response(getattr(response, "text", ""))
            logger.info("*************** GEMINI SUGGESTIONS (RAW) ***************")
            logger.info(json.dumps(suggestions, indent=2))
            logger.info("********************************************************")
            
            # Validate and fix suggestions to prevent text operations on categorical columns
            suggestions = self._validate_and_fix_suggestions(suggestions, inspection_data)
            logger.info("*************** GEMINI SUGGESTIONS (VALIDATED) ***************")
            logger.info(json.dumps(suggestions, indent=2))
            logger.info("**************************************************************")
            
            inspection_data["suggestions"] = suggestions
            logger.info(f"Gemini analysis completed for {len(suggestions)} columns")
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}, falling back to heuristic")
            inspection_data["suggestions"] = self._heuristic_analysis(inspection_data)["suggestions"]

        return inspection_data

    def _safe_generate_content(self, prompt: str, max_retries: int = 3):
        """Retry Gemini if rate limit exceeded."""
        for attempt in range(max_retries):
            try:
                return self.model.generate_content(prompt)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Quota exceeded. Waiting {wait_time}s (attempt {attempt+1})")
                    time.sleep(wait_time)
                else:
                    raise
        raise RuntimeError("Failed to get response from Gemini after retries.")

    def _create_analysis_prompt(self, inspection_data: Dict[str, Any]) -> str:
        """Prompt Gemini to return multiple actions per column with professional data engineering approach."""
        profile = inspection_data["profile"]
        samples = inspection_data["samples"]

        prompt = """You are a Senior Data Engineer with expertise in data preprocessing and feature engineering. 
Analyze this dataset and suggest a comprehensive preprocessing pipeline for each column.

CRITICAL: A single column may require MULTIPLE sequential operations. For example:
- A column with outliers AND wide range → ["remove_outliers", "standardize"]
- A text column with missing values → ["fill_missing", "text_clean"]
- A numeric column with nulls and outliers → ["fill_missing", "remove_outliers", "normalize_range"]

Dataset Profile:
"""

        for col, info in profile.items():
            prompt += f"\nColumn '{col}':\n"
            prompt += f"  Type: {info['dtype']}\n"
            prompt += f"  Non-null: {info['non_null']}, Nulls: {info['nulls']}\n"
            if 'min' in info:
                prompt += f"  Numeric stats: min={info['min']}, max={info['max']}, mean={info['mean']:.2f}\n"
                # Calculate potential outlier indicators
                if info['min'] is not None and info['max'] is not None:
                    range_val = info['max'] - info['min']
                    mean_val = info.get('mean', 0)
                    if mean_val > 0:
                        cv = (range_val / mean_val) * 100 if mean_val != 0 else 0
                        prompt += f"  Range: {range_val:.2f}, Coefficient of Variation: {cv:.1f}%\n"
            else:
                prompt += f"  Unique values: {info['unique_count']}\n"
                if info.get('sample_unique'):
                    prompt += f"  Sample values: {info['sample_unique'][:5]}\n"
            if col in samples:
                prompt += f"  Sample data: {samples[col][:3]}\n"

        
        prompt += """
 
Analyze the provided dataset and output ONLY valid JSON. Do not include any prose, explanation, or extra text — only JSON.

1) Detect and remove useless columns:
   - Automatically detect and exclude identifier-like columns (exact or fuzzy matches to: "id", "ID", "Id", "name", "Name", "serial", "serial_no", "index", "uid", "unique", "uuid", "row_id", "sno"). Do NOT include these columns in the output JSON.

2) Column typing:
   - For every remaining column determine its type: numeric, categorical, text, datetime.
   - If type is ambiguous, infer using common heuristics (numeric parseable → numeric, many unique strings and natural language → text, ISO-like patterns → datetime, small set of repeated strings → categorical).
   - IMPORTANT: Short categorical columns (gender, status, category, type, class, etc.) with few unique values (<10) and short average length (<20 chars) should be treated as CATEGORICAL, NOT text. Do NOT apply text_clean, remove_stopwords, lemmatize, or tfidf to these columns.

3) OPERATION SEQUENCE (Agents MUST be applied in this exact order for every column):
   1. fix_infinite
   3. remove_outliers
   2. fill_missing
   4. deduplicate
   5. drop_constant
   6. handle_datetime
   7. handle_skewness
   8. categorical_encode
   9. text_clean
   10. remove_stopwords
   11. lemmatize
   12. tfidf
   13. standardize_or_normalize_range

   - You MUST iterate through every agent in this order for each useful column.
   - If an agent is not applicable for a given column, output the agent with suggestion "skip" and an appropriate short reason.
   - The final agent (13) must **never** be ignored: include either "standardize" OR "normalize_range" for every numeric or numeric-ready column. 
   - If the column is purely textual and will be converted to TF-IDF, include "standardize": "skip" with a reason.
   - Double-check to ensure the final scaling step is always present for numeric/numeric-ready columns before returning output.

4) Rules and agent behaviors (brief):
   - fix_infinite: replace +/-inf with NaN or a valid numeric placeholder.
   - fill_missing: impute missing values based on type and distribution (numeric: mean/median; categorical: mode; text/datetime: sentinel or skip). Include parameters for chosen method.
   - remove_outliers: detect and handle outliers using IQR or Z-score methods. Add method name and threshold in parameters.
   - deduplicate: remove duplicate rows or values. Provide params: {"subset": [<fields>], "keep": "first"}.
   - drop_constant: drop columns with a single unique value or very low variance (threshold param optional).
   - handle_datetime: parse and optionally extract features (year, month, day, weekday, is_weekend, quarter). If not datetime, skip.
   - handle_skewness: compute skewness; if abs(skewness) > 1.0 (configurable), apply log1p or Box-Cox; else skip.
   - categorical_encode: for categorical columns, choose onehot if cardinality <= 10 else label encoding. Provide parameters.
   - text_clean: lowercase, remove special chars, numbers, and extra spaces. ONLY for long-form text columns (avg length > 20 chars). For short categorical columns (gender, status, etc.), use "skip" with reason "short categorical column".
   - remove_stopwords: remove common stopwords from text. ONLY for long-form text. Skip for categorical columns.
   - lemmatize: reduce words to their base form. ONLY for long-form text. Skip for categorical columns.
4b) NLP vs CATEGORICAL ENFORCEMENT:
   - If a column is NLP (long-form text), you MUST include these steps in the defined order for that column:
       ["text_clean", "remove_stopwords", "lemmatize", "tfidf"].
   - If a column is NOT NLP (i.e., short categorical text), you MUST set these four steps to {"suggestion": "skip"} with a concise reason like "short categorical column" or "not NLP".
   - Do NOT apply text steps to categorical columns. Prefer "categorical_encode" for such columns.

4c) CATEGORICAL ENCODING ENFORCEMENT:
   - If a column is CATEGORICAL (non-NLP short text), you MUST include the action "categorical_encode" (do not skip it).
   - Choose strategy by cardinality: if unique_count <= 10 → "onehot" else → "label". Provide parameters accordingly.
   - Do NOT propose text steps for categorical columns (enforced above). If model mistakenly suggests them, set them to "skip" with reason.

5) Output schema (exact JSON format):
{
  "<column_name>": [
    {"suggestion": "<action_name>", "reason": "<short reason>", "parameters": { ... }},
    ...
  ],
  ...
}

- For every useful column include exactly one entry for each agent in the OPERATION SEQUENCE (1→13) in the same order.
- When agent is not applicable, set "suggestion" to "skip" and provide a concise "reason".
- For the final agent use one of:
  {"suggestion":"standardize", ...} OR {"suggestion":"normalize_range", ...} OR {"suggestion":"skip", "reason":"<why (only allowed for non-numeric such as pure text after tfidf)>"}.
- If you choose "tfidf" for a text column, then for the final scaling step explain whether scaling is necessary; prefer "skip" for scaling TF-IDF vectors with reason.

6) Validation checks (do these automatically and report only via JSON fields):
   - Confirm there are no identifier-like columns included.
   - Confirm each useful column has 13 agent entries in the correct order.
   - Confirm final agent is present for each column; if missing, auto-add it and mark {"parameters": {"auto_fixed": true}}.
   - Confirm that categorical columns include a "categorical_encode" step (with strategy). If missing, auto-add it with a justified strategy and mark {"parameters": {"auto_fixed": true}}.
   - If drop_constant is recommended for a column, still include the subsequent agents but mark them with "skip" where logically unnecessary (e.g., after a drop_constant recommendation include skip for handle_skewness with reason "column will be dropped").

7) Minimal verbosity:
   - The JSON must be machine-parseable. Keep each "reason" concise (max 1–2 short sentences). Parameters should be small dictionaries with explicit keys.

8) AVAILABLE ACTIONS (use these exact action strings):
AVAILABLE_ACTIONS = 
    "fix_infinite",
    "remove_outliers",
    "fill_missing",          
    "deduplicate",
    "drop_constant",
    "handle_datetime",
    "handle_skewness",
    "categorical_encode",
    "text_clean",
    "remove_stopwords",
    "lemmatize",
    "tfidf",
    "standardize",
    "normalize_range",
    "skip"



8b) AGENT CAPABILITIES AND PARAM SCHEMAS (use exactly these parameter keys):
   - fix_infinite:
       parameters: { }
       notes: Replaces +/-inf with NaN for numeric columns.
   - fill_missing:
       parameters: { "strategy": "auto" | "mean" | "median" | "mode" | "forward_fill" | "backward_fill" }
       notes: If omitted, orchestrator may set defaults; prefer an explicit strategy.
   - remove_outliers:
       parameters: { "method": "remove_outliers", "outlier_strategy": "median" | "clip" | "nan" }
       notes: Uses IQR bounds; choose how to handle detected outliers.
   - deduplicate:
       parameters: { "subset": ["<col>", ...] | null, "keep": "first" | "last" | false }
       notes: If subset is null, duplicates are removed across all columns.
   - drop_constant:
       parameters: { }
       notes: Drops columns with <= 1 unique value.
   - handle_datetime:
       parameters: { }
       notes: Parses to datetime and extracts year, month, day, weekday, hour; drops original.
   - handle_skewness:
       parameters: { }
       notes: If |skew| > 1, applies log1p (shifted if non-positive values present).
   - categorical_encode:
       parameters: { "method": "categorical_encode", "strategy": "onehot" | "label", "keep_original": bool }
       notes:
         • Do NOT invent new action names like "onehot" or "label_encode". Use action "categorical_encode" with strategy set accordingly.
   - text_clean:
       parameters: { "operations": [
           "lowercase",
           "remove_special_chars",
           "remove_stopwords",
           "strip_whitespace",
           "remove_extra_spaces",
           "remove_numbers"
         ] }
       notes:
         • Choose a minimal sensible subset based on data (e.g., ["lowercase","remove_special_chars","strip_whitespace"]).
   - remove_stopwords:
       parameters: { }
       notes: Only for long-form text columns.
   - lemmatize:
       parameters: { }
       notes: Only for long-form text columns.
   - tfidf:
       parameters: { "max_features": <int, default 100> }
       notes: Replaces the original text column with TF-IDF features.
   - standardize:
       parameters: { "method": "standard" }
   - normalize_range:
       parameters: { "method": "minmax" }

Now, analyze the dataset columns, remove useless identifier-like columns, and produce the JSON described above, iterating every agent in the defined order for each useful column. BEFORE returning, run an internal check to ensure the final scaling agent (standardize or normalize_range) is present for each numeric or numeric-ready column; if absent, auto-add it and mark {"parameters": {"auto_fixed": true}}.
"""
       
        



        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse JSON safely for multiple actions per column."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object in case of extra text
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except Exception as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
            logger.error("No valid JSON found in Gemini response")
            return {}
    
    def _validate_and_fix_suggestions(self, suggestions: Dict[str, List[Dict[str, Any]]], 
                                     inspection_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate and fix Gemini suggestions to prevent text operations on categorical columns.
        This is a hard enforcement layer that overrides Gemini if it suggests incorrect operations.
        """
        from .nlp_utils import is_nlp_column
        
        profile = inspection_data.get("profile", {})
        fixed_suggestions = {}
        
        for col, actions in suggestions.items():
            if col not in profile:
                fixed_suggestions[col] = actions
                continue
            
            col_info = profile[col]
            fixed_actions = []
            
            # Determine if this is truly an NLP column
            try:
                # Get sample data for this column
                if "samples" in inspection_data and col in inspection_data["samples"]:
                    sample_series = pd.Series(inspection_data["samples"][col])
                else:
                    # Create a dummy series from profile info
                    sample_values = col_info.get("sample_unique", [""])
                    sample_series = pd.Series(sample_values * 10)  # Repeat to get enough samples
                
                is_nlp = is_nlp_column(col, sample_series)
            except Exception as e:
                logger.warning(f"Failed to check if '{col}' is NLP column: {e}, assuming not NLP")
                is_nlp = False
            
            # Check if it's a short categorical column
            is_categorical = (
                col_info.get("dtype") == "object" and
                col_info.get("unique_count", 999) < 20 and
                not is_nlp
            )
            
            text_operations = {"text_clean", "remove_stopwords", "lemmatize", "tfidf"}
            
            for action in actions:
                suggestion = action.get("suggestion", "")
                
                # Force skip text operations on categorical columns
                if is_categorical and suggestion in text_operations:
                    logger.warning(
                        f"🔧 OVERRIDING Gemini: Column '{col}' is categorical (unique={col_info.get('unique_count')}), "
                        f"forcing '{suggestion}' to 'skip'"
                    )
                    fixed_actions.append({
                        "suggestion": "skip",
                        "reason": f"Short categorical column (was: {suggestion})",
                        "parameters": {}
                    })
                else:
                    fixed_actions.append(action)
            
            fixed_suggestions[col] = fixed_actions
        
        return fixed_suggestions

    def _heuristic_analysis(self, inspection_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Advanced heuristic analysis: multiple sequential steps per column with proper ordering."""
        suggestions = {}
        profile = inspection_data["profile"]

        for col, info in profile.items():
            col_steps = []
            
            # Step 1: Handle missing values first (data quality)
            if info["nulls"] > 0:
                null_percentage = (info["nulls"] / (info["non_null"] + info["nulls"])) * 100
                col_steps.append({
                    "suggestion": "fill_missing",
                    "reason": f"{info['nulls']} missing values ({null_percentage:.1f}% of data)",
                    "parameters": {}
                })
            
            # Step 2: Handle outliers for numeric columns (after filling missing values)
            if 'min' in info and info['min'] is not None and info['max'] is not None:
                range_val = info['max'] - info['min']
                mean_val = info.get('mean', 0)
                
                # Check for potential outliers using coefficient of variation
                if mean_val > 0:
                    cv = (range_val / mean_val) * 100
                    if cv > 200:  # High coefficient of variation suggests outliers
                        col_steps.append({
                            "suggestion": "remove_outliers",
                            "reason": f"High coefficient of variation ({cv:.1f}%) suggests outliers in range [{info['min']:.2f}, {info['max']:.2f}]",
                            "parameters": {}
                        })
                
                # Step 3: Scaling/normalization (after outlier removal)
                if range_val > 1000:  # Wide range
                    col_steps.append({
                        "suggestion": "normalize_range",
                        "reason": f"Wide numeric range ({range_val:.2f}) requires scaling for ML algorithms",
                        "parameters": {}
                    })
                elif range_val > 100:  # Moderate range - use standardization
                    col_steps.append({
                        "suggestion": "standardize",
                        "reason": f"Moderate range ({range_val:.2f}) - standardization recommended for better ML performance",
                        "parameters": {}
                    })
            
            # Step 4: Text processing for categorical/text columns
            elif info['dtype'] == 'object':
                if info['unique_count'] > 10:
                    col_steps.append({
                        "suggestion": "text_clean",
                        "reason": f"Text column with {info['unique_count']} unique values needs cleaning and normalization",
                        "parameters": {}
                    })
                elif info['unique_count'] <= 10:
                    col_steps.append({
                        "suggestion": "categorical_encode",
                        "reason": f"Low cardinality categorical column ({info['unique_count']} categories) - encoding recommended",
                        "parameters": {}
                    })
            
            # If no specific preprocessing needed
            if not col_steps:
                col_steps.append({
                    "suggestion": "skip",
                    "reason": "Column appears clean and ready for analysis",
                    "parameters": {}
                })

            suggestions[col] = col_steps

        return {"suggestions": suggestions}


# below code is for single agent per column , above code is for multiple agent per column
# import json
# import time
# from typing import Dict, Any, Optional

# import google.generativeai as genai
# import pandas as pd

# try:
#     from .inspector_agent import InspectorAgent
#     from ...config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
#     from ...utils.logger import get_logger
# except ImportError:
#     from agents.inspector.inspector_agent import InspectorAgent
#     from config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
#     from utils.logger import get_logger


# logger = get_logger(__name__)


# class GeminiInspectorAgent(InspectorAgent):
#     """Inspector agent that uses Gemini to analyze data samples and suggest preprocessing steps."""

#     def __init__(self):
#         super().__init__()
#         self.name = "GeminiInspectorAgent"

#         if not GOOGLE_API_KEY:
#             logger.warning("GOOGLE_API_KEY not set, falling back to heuristic inspection")
#             self.gemini_available = False
#         else:
#             genai.configure(api_key=GOOGLE_API_KEY)
#             self.gemini_available = True
#             self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

#     def process(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
#         """Analyze dataframe using Gemini and return suggestions."""
#         # Get base inspection data
#         inspection_data = super().process(df, **kwargs)

#         if not self.gemini_available:
#             return self._heuristic_analysis(inspection_data)

#         try:
#             # Prepare prompt for Gemini
#             prompt = self._create_analysis_prompt(inspection_data)

#             # Get Gemini's analysis with retry logic
#             response = self._safe_generate_content(prompt)
#             suggestions = self._parse_gemini_response(response.text)

#             inspection_data["suggestions"] = suggestions
#             logger.info(f"Gemini analysis completed with {len(suggestions)} suggestions")

#         except Exception as e:
#             logger.error(f"Gemini analysis failed: {e}, falling back to heuristic")
#             inspection_data["suggestions"] = self._heuristic_analysis(inspection_data)["suggestions"]

#         return inspection_data

#     def _safe_generate_content(self, prompt: str, max_retries: int = 3):
#         """Handle Gemini rate limits (429) by retrying with exponential backoff."""
#         for attempt in range(max_retries):
#             try:
#                 return self.model.generate_content(prompt)
#             except Exception as e:
#                 error_message = str(e)
#                 if "429" in error_message or "quota" in error_message.lower():
#                     wait_time = (attempt + 1) * 10
#                     logger.warning(f"Quota exceeded. Waiting {wait_time}s before retrying... (attempt {attempt+1})")
#                     time.sleep(wait_time)
#                     continue
#                 else:
#                     raise
#         raise RuntimeError("Failed to get response from Gemini after multiple retries.")

#     def _create_analysis_prompt(self, inspection_data: Dict[str, Any]) -> str:
#         """Create a structured prompt for Gemini analysis."""
#         profile = inspection_data["profile"]
#         samples = inspection_data["samples"]

#         prompt = f"""
# You are a data preprocessing expert. Analyze this dataset sample and suggest preprocessing steps for each column.

# Dataset Profile:
# """

#         for col, info in profile.items():
#             prompt += f"\nColumn '{col}':\n"
#             prompt += f"  Type: {info['dtype']}\n"
#             prompt += f"  Non-null: {info['non_null']}, Nulls: {info['nulls']}\n"

#             if 'min' in info:
#                 prompt += f"  Numeric stats: min={info['min']}, max={info['max']}, mean={info['mean']:.2f}\n"
#             else:
#                 prompt += f"  Unique values: {info['unique_count']}\n"
#                 if info.get('sample_unique'):
#                     prompt += f"  Sample values: {info['sample_unique'][:5]}\n"

#             if col in samples:
#                 prompt += f"  Sample data: {samples[col][:3]}\n"

#         prompt += """
# Suggest preprocessing steps for each column. Return ONLY a JSON object with this structure:
# {
#   "column_name": {
#     "suggestion": "action_name",
#     "reason": "brief explanation",
#     "parameters": {}
#   }
# }

# Available actions:
# - "fill_missing": for missing value imputation
# - "normalize_range": for numeric scaling to [0,1]
# - "standardize": for z-score normalization
# - "remove_outliers": for outlier detection/removal
# - "deduplicate": for removing duplicates
# - "text_clean": for text preprocessing
# - "categorical_encode": for one-hot or label encoding
# - "skip": if no preprocessing needed

# Focus on the most impactful preprocessing steps. Be conservative - only suggest what's clearly needed.
# """
#         return prompt

#     def _parse_gemini_response(self, response_text: str) -> Dict[str, Dict[str, Any]]:
#         """Parse Gemini's JSON response."""
#         try:
#             # Extract JSON from response (in case there's extra text)
#             start = response_text.find('{')
#             end = response_text.rfind('}') + 1
#             if start >= 0 and end > start:
#                 json_str = response_text[start:end]
#                 return json.loads(json_str)
#             else:
#                 raise ValueError("No valid JSON found in response")
#         except Exception as e:
#             logger.error(f"Failed to parse Gemini response: {e}")
#             return {}

#     def _heuristic_analysis(self, inspection_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
#         """Fallback heuristic analysis when Gemini is unavailable."""
#         suggestions = {}
#         profile = inspection_data["profile"]

#         for col, info in profile.items():
#             suggestions[col] = {"suggestion": "skip", "reason": "No preprocessing needed"}

#             # Check for missing values
#             if info["nulls"] > 0:
#                 suggestions[col] = {
#                     "suggestion": "fill_missing",
#                     "reason": f"Found {info['nulls']} missing values",
#                     "parameters": {}
#                 }

#             # Check for numeric columns that might need scaling
#             elif 'min' in info and info['min'] is not None:
#                 min_val, max_val = info['min'], info['max']
#                 if max_val - min_val > 1000:  # Wide range
#                     suggestions[col] = {
#                         "suggestion": "normalize_range",
#                         "reason": f"Wide numeric range ({min_val:.2f} to {max_val:.2f})",
#                         "parameters": {}
#                     }

#             # Check for text columns
#             elif info['dtype'] == 'object' and info['unique_count'] > 10:
#                 suggestions[col] = {
#                     "suggestion": "text_clean",
#                     "reason": "Text column with many unique values",
#                     "parameters": {}
#                 }

#         return {"suggestions": suggestions}
