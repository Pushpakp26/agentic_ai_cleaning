# agents/inspector/gemini_pyspark_inspector_agent.py
import json
import time
from typing import Dict, Any, List
import google.generativeai as genai
from pyspark.sql import DataFrame

try:
    from .inspector_pyspark import PySparkInspectorAgent
    from ...config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
    from ...utils.logger import get_logger
except ImportError:
    from agents.inspector.inspector_pyspark import PySparkInspectorAgent
    from config import GEMINI_MODEL_NAME, GOOGLE_API_KEY
    from utils.logger import get_logger


logger = get_logger(__name__)


class GeminiInspectorAgentSpark(PySparkInspectorAgent):
    """Gemini-powered Inspector Agent for PySpark DataFrames."""

    def __init__(self, spark=None, df=None):
        super().__init__(spark, df)
        self.name = "GeminiInspectorAgentSpark"

        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set, falling back to heuristic inspection (Spark)")
            self.gemini_available = False
        else:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            self.gemini_available = True

    def process(self, df: DataFrame, **kwargs) -> Dict[str, Any]:
        """Main entry point — calls parent inspector for samples and adds Gemini suggestions."""
        # ✅ This is the correct hierarchical call (same as pandas)
        inspection_data = super().process(df)

        try:
            if not self.gemini_available:
                logger.info("Gemini not available — using heuristic Spark analysis")
                return self._heuristic_analysis(inspection_data)

            # Use Gemini for AI-based suggestions
            prompt = self._create_analysis_prompt(inspection_data)
            # Log the prompt for observability
            logger.debug("*************** GEMINI PROMPT (SPARK) ***************\n%s\n***************************************************", prompt)
            response = self._safe_generate_content(prompt)
            # Log raw Gemini response text to console
            try:
                logger.info("GEMINI RAW RESPONSE (SPARK): %s", getattr(response, "text", str(response)))
            except Exception as log_exc:
                logger.warning(f"Failed to log raw Gemini response: {log_exc}")
            suggestions = self._parse_gemini_response(getattr(response, "text", ""))
            inspection_data["suggestions"] = suggestions
            logger.info(f"Gemini Spark analysis completed for {len(suggestions)} columns")

        except Exception as e:
            logger.error(f"Gemini Spark analysis failed: {e}, using heuristic fallback")
            inspection_data["suggestions"] = self._heuristic_analysis(inspection_data)["suggestions"]

        return inspection_data

    # ---------------------------------------
    # Gemini Logic (unchanged)
    # ---------------------------------------

    def _safe_generate_content(self, prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return self.model.generate_content(prompt)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Quota exceeded. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
        raise RuntimeError("Gemini API failed after retries.")

    def _create_analysis_prompt(self, inspection_data: Dict[str, Any]) -> str:
        profile = inspection_data["profile"]
        samples = inspection_data["samples"]

        prompt = "You are a data preprocessing expert. Analyze this PySpark dataset and suggest preprocessing steps.\n\nDataset Profile:\n"

        for col, info in profile.items():
            prompt += f"\nColumn '{col}':\n"
            prompt += f"  Type: {info['dtype']}\n"
            prompt += f"  Non-null: {info['non_null']}, Nulls: {info['nulls']}\n"
            if 'min' in info:
                prompt += f"  Numeric stats: min={info['min']}, max={info['max']}, mean={info['mean']:.2f}\n"
            else:
                prompt += f"  Unique values: {info.get('unique_count', 0)}\n"
            if col in samples:
                prompt += f"  Sample data: {samples[col][:3]}\n"

        prompt +="""
Analyze the provided dataset and output ONLY valid JSON. Do not include any prose, explanation, or extra text — only JSON.

1) Detect and remove useless columns:
   - Automatically detect and exclude identifier-like columns (exact or fuzzy matches to: "id", "ID", "Id", "name", "Name", "serial", "serial_no", "index", "uid", "unique", "uuid", "row_id", "sno"). Do NOT include these columns in the output JSON.

2) Column typing:
   - For every remaining column determine its type: numeric, categorical, text, datetime.
   - If type is ambiguous, infer using common heuristics (numeric parseable → numeric, many unique strings and natural language → text, ISO-like patterns → datetime, small set of repeated strings → categorical).

3) OPERATION SEQUENCE (Agents MUST be applied in this exact order for every column):
   1. fill_missing
   2. fix_infinite
   3. deduplicate
   4. drop_constant
   5. handle_datetime
   6. handle_skewness
   7. categorical_encode
   8. text_clean
   9. remove_stopwords
   10. lemmatize
   11. tfidf
   12. standardize_or_normalize_range

   - You MUST iterate through every agent in this order for each useful column.
   - If an agent is not applicable for a given column, output the agent with suggestion "skip" and an appropriate short reason.
   - The final agent (12) must **never** be ignored: include either "standardize" OR "normalize_range" for every numeric or numeric-ready column. If the column is purely textual and will be converted to TF-IDF, include `"standardize": "skip"` with a reason. Still perform the check twice before producing output to ensure final scaling agent is present as required.

4) Rules and agent behaviors (brief):
   - fill_missing: choose method based on type and distribution (numeric: mean/median; categorical: mode; text/datetime: skip or fill sentinel). If imputed, document method in parameters.
   - fix_infinite: convert +/-inf to NaN or a numeric placeholder, then mark for re-imputation.
   - deduplicate: recommend deduplicate at row-level when duplicate rows exist. Provide params: {"subset": [<fields>]}. If column-level duplicates only, use subset with that column.
   - drop_constant: drop when unique_count <= 1 or near-constant (threshold param optional).
   - handle_datetime: parse and optionally extract features like year, month, day, weekday, is_weekend, quarter. If not datetime, skip.
   - handle_skewness: compute skewness; if abs(skewness) > 1.0 (configurable) apply log1p or Box-Cox; otherwise skip.
   - categorical_encode: for categorical columns choose onehot if cardinality <= threshold (default 10), else label encoding. Provide parameters.
   - text_clean → remove_stopwords → lemmatize → tfidf: always follow this pipeline for text columns. If text column is very short or a label, explain skip.
   - standardize_or_normalize_range: For numeric features choose "standardize" (z-score) by default; if explicitly requested for algorithms needing [0,1], choose "normalize_range". Always include this agent in the output for numeric/numeric-ready columns; do a double-check validation before finishing.

5) Output schema (exact JSON format):
{
  "<column_name>": [
    {"suggestion": "<action_name>", "reason": "<short reason>", "parameters": { ... }},
    ...
  ],
  ...
}

- For every useful column include **exactly one entry for each agent in the OPERATION SEQUENCE (1→12)** in the same order.
- When agent is not applicable, set "suggestion" to "skip" and provide a concise "reason".
- For the final agent use either: {"suggestion":"standardize", ...} OR {"suggestion":"normalize_range", ...} OR {"suggestion":"skip", "reason":"<why (only allowed for non-numeric such as pure text after tfidf)>"}.
- If you choose "tfidf" for a text column, then for the final scaling step explain whether scaling is necessary; prefer "skip" for scaling TF-IDF vectors with reason.

6) Validation checks (do these automatically and report only via JSON fields):
   - Confirm there are no identifier-like columns included.
   - Confirm each useful column has 12 agent entries in the correct order.
   - Confirm final agent is present for each column; if missing, add it and note in parameters: {"auto_fixed": true}.
   - If drop_constant is recommended for a column, still include the subsequent agents but mark them with "skip" where logically unnecessary (e.g., after a drop_constant recommendation include skip for handle_skewness with reason "column will be dropped").

7) Minimal verbosity:
   - The JSON must be machine-parseable. Keep each "reason" concise (max 1–2 short sentences). Parameters should be small dictionaries with explicit keys.

8) AVAILABLE ACTIONS (use these exact action strings):
- "fill_missing"
- "fix_infinite"
- "deduplicate"
- "drop_constant"
- "handle_datetime"
- "handle_skewness"
- "categorical_encode"
- "text_clean"
- "remove_stopwords"
- "lemmatize"
- "tfidf"
- "standardize"  (use for z-score)
- "normalize_range" (use for MinMax)
- "skip"

Now, analyze the dataset columns, remove useless identifier-like columns, and produce the JSON described above, iterating every agent in the defined order for each useful column. BEFORE returning, run an internal check to ensure the final scaling agent (standardize or normalize_range) is present for each numeric or numeric-ready column; if absent, auto-add it and mark {"parameters": {"auto_fixed": true}}.
"""

        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except Exception as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
            logger.error("No valid JSON found in Gemini response")
            return {}

    def _heuristic_analysis(self, inspection_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        profile = inspection_data["profile"]
        suggestions = {}

        for col, info in profile.items():
            steps = []
            if info["nulls"] > 0:
                steps.append({"suggestion": "fill_missing", "reason": f"{info['nulls']} missing values"})
            if "min" in info and "max" in info and info["max"] - info["min"] > 1000:
                steps.append({"suggestion": "normalize_range", "reason": "Wide numeric range"})
            if "string" in info["dtype"].lower() and info.get("unique_count", 0) > 10:
                steps.append({"suggestion": "text_clean", "reason": "High cardinality text"})
            if not steps:
                steps.append({"suggestion": "skip", "reason": "No preprocessing required"})
            suggestions[col] = steps

        return {"suggestions": suggestions}
