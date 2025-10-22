from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import pandas as pd
from pyspark.sql import DataFrame

try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)


class InspectorInterface(ABC):
    """Interface for data inspection agents that return analysis results."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, df: Union[pd.DataFrame, DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Analyze the dataframe and return inspection results.
        
        Args:
            df: Input DataFrame (Pandas or PySpark)
            **kwargs: Additional parameters for inspection
            
        Returns:
            Dictionary containing inspection results and suggestions
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this inspector's capabilities."""
        return {
            "inspector_name": self.name,
            "interface_type": "inspector",
            "return_type": "Dict[str, Any]"
        }


class ProcessingAgentInterface(ABC):
    """Interface for data processing agents that return transformed DataFrames."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, df: Union[pd.DataFrame, DataFrame], **kwargs) -> Union[pd.DataFrame, DataFrame]:
        """
        Process the dataframe and return the transformed result.
        
        Args:
            df: Input DataFrame (Pandas or PySpark)
            **kwargs: Additional parameters for processing
            
        Returns:
            Transformed DataFrame of the same type as input
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this agent's processing capabilities."""
        return {
            "agent_name": self.name,
            "interface_type": "processor",
            "return_type": "DataFrame"
        }
