import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

from agents.base_agent import BaseAgent
from agents.inspector.nlp_utils import is_nlp_column
from utils.logger import get_logger

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
lemmatizer = WordNetLemmatizer()
logger = get_logger(__name__)

class LemmatizeTextAgent(BaseAgent):
    """Agent to lemmatize long text columns."""

    def __init__(self):
        super().__init__("LemmatizeTextAgent")

    def process(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        if not is_nlp_column(column, df[column]):
            return df
        df[column] = df[column].astype(str).apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()]))
        logger.info(f"Lemmatized text in '{column}'")
        return df
