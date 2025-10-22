import pandas as pd

def is_nlp_column(col_name: str, sample_text: pd.Series) -> bool:
    """
    Determine if a column contains NLP text (long-form text requiring text processing).
    
    Returns False for:
    - Identifier columns (id, name, etc.)
    - Short categorical columns (gender, status, category, type, etc.)
    - Columns with average length <= 20 characters
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
    
    # Check average text length
    try:
        avg_len = sample_text.dropna().astype(str).apply(len).mean()
    except Exception:
        return False
    
    # Only consider columns with average length > 20 as NLP text
    # Also check unique count - if very few unique values, it's likely categorical
    if avg_len <= 20:
        return False
    
    # Additional check: if unique count is very low relative to total, it's categorical
    try:
        unique_ratio = sample_text.nunique() / len(sample_text)
        if unique_ratio < 0.05:  # Less than 5% unique values suggests categorical
            return False
    except Exception:
        pass
    
    return True
