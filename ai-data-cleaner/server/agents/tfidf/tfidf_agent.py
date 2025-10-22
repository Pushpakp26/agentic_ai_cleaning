import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from agents.base_agent import BaseAgent
from agents.inspector.nlp_utils import is_nlp_column
from utils.logger import get_logger

logger = get_logger(__name__)

class TFIDFAgent(BaseAgent):
    """Agent to convert long text columns into TF-IDF features."""

    def __init__(self):
        super().__init__("TFIDFAgent")

    def process(self, df: pd.DataFrame, column: str, max_features: int = 100, **kwargs) -> pd.DataFrame:
        if not is_nlp_column(column, df[column]):
            return df
        tfidf = TfidfVectorizer(max_features=max_features)
        vectors = tfidf.fit_transform(df[column].astype(str).fillna("")).toarray()
        feature_names = [f"{column}_tfidf_{i}" for i in range(vectors.shape[1])]
        tfidf_df = pd.DataFrame(vectors, columns=feature_names, index=df.index)
        df = pd.concat([df.drop(columns=[column]), tfidf_df], axis=1)
        logger.info(f"TF-IDF vectorized '{column}' into {vectors.shape[1]} features")
        return df
