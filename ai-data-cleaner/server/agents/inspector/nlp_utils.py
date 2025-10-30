import pandas as pd

def is_nlp_column(col_name: str, sample_text: pd.Series) -> bool:
    """
    Determine if a column contains NLP text (long-form text requiring text processing).
    
    Returns False for:
    - Identifier columns (id, name, etc.)
    - Short categorical columns (gender, status, category, type, etc.)
    - Columns with average length <= 20 characters or average words <= 3
    - Numeric or mostly numeric columns
    """
    low_name = col_name.lower()
    
    # Exclude identifier-like columns
    if any(x in low_name for x in ["name", "id", "location", "city", "country", "state", "address"]):
        return False
    
    # Exclude common categorical column names
    categorical_keywords = [
        "gender", "sex", "status", "category", "type", "class", "label",
        "group", "level", "grade", "rank", "tier", "flag", "indicator",
        "code", "tag", "segment"
    ]
    if any(keyword in low_name for keyword in categorical_keywords):
        return False

    # Early exit for numeric or mostly numeric columns
    if pd.api.types.is_numeric_dtype(sample_text):
        return False
    try:
        # If most values can be converted to numbers, it's not text
        numeric_ratio = sample_text.dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).mean()
        if numeric_ratio > 0.8:  # 80% numeric-like entries
            return False
    except Exception:
        pass

    # Check average text length and average word count
    try:
        cleaned = sample_text.dropna().astype(str)
        avg_len = cleaned.apply(len).mean()
        avg_words = cleaned.apply(lambda x: len(x.split())).mean()
    except Exception:
        return False

    # Reject short or low-word columns
    if avg_len <= 15 or avg_words <= 3:
        return False

    # Check uniqueness ratio to detect categorical-like columns
    try:
        unique_ratio = sample_text.nunique() / len(sample_text)
        if unique_ratio < 0.05:  # Less than 5% unique values suggests categorical
            return False
    except Exception:
        pass

    return True
