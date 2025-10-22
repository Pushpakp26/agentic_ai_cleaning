import pandas as pd
import nltk
from nltk.corpus import stopwords
from agents.base_agent import BaseAgent
from agents.inspector.nlp_utils import is_nlp_column
from utils.logger import get_logger

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
logger = get_logger(__name__)

class RemoveStopwordsAgent(BaseAgent):
    """Agent to remove stopwords from long text columns."""

    def __init__(self):
        super().__init__("RemoveStopwordsAgent")

    def process(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        if not is_nlp_column(column, df[column]):
            return df
        df[column] = df[column].astype(str).apply(lambda x: " ".join([w for w in x.split() if w.lower() not in stop_words]))
        logger.info(f"Removed stopwords from '{column}'")
        return df
