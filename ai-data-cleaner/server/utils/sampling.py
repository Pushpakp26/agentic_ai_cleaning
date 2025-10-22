from __future__ import annotations

from typing import Dict, Any

import pandas as pd

try:
    from ..config import SAMPLE_ROWS_PER_COLUMN, MAX_UNIQUE_PREVIEW
except ImportError:
    from config import SAMPLE_ROWS_PER_COLUMN, MAX_UNIQUE_PREVIEW


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
	"""Return simple per-column profile stats to accompany samples."""
	profile: Dict[str, Dict[str, Any]] = {}
	for col in df.columns:
		series = df[col]
		info: Dict[str, Any] = {
			"dtype": str(series.dtype),
			"non_null": int(series.notna().sum()),
			"nulls": int(series.isna().sum()),
		}
		if pd.api.types.is_numeric_dtype(series):
			info.update(
				{
					"min": try_float(series.min()),
					"max": try_float(series.max()),
					"mean": try_float(series.mean()),
					"std": try_float(series.std()),
				}
			)
		else:
			uniq_vals = series.dropna().unique()
			info.update(
				{
					"unique_count": int(len(uniq_vals)),
					"sample_unique": [str(v) for v in uniq_vals[:MAX_UNIQUE_PREVIEW]],
				}
			)
		profile[col] = info
	return profile


def sample_per_column(df: pd.DataFrame, rows: int = SAMPLE_ROWS_PER_COLUMN) -> Dict[str, Any]:
	"""Collect up to N non-null samples per column alongside profiles."""
	samples: Dict[str, Any] = {}
	for col in df.columns:
		series = df[col].dropna()
		samples[col] = series.sample(n=min(len(series), rows), random_state=42).tolist()
	return {"profile": profile_dataframe(df), "samples": samples}


def try_float(value):
	try:
		return float(value)
	except Exception:
		return None


