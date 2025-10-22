import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

try:
    from ..base_agent import BaseAgent
    from ...utils.logger import get_logger
except ImportError:
    from agents.base_agent import BaseAgent
    from utils.logger import get_logger

logger = get_logger(__name__)

class EncodingAgent(BaseAgent):
    """Agent for encoding categorical columns using scikit-learn encoders."""

    def __init__(self):
        super().__init__("EncodingAgent")

    def process(
        self,
        df: pd.DataFrame,
        column: str = None,
        method: str = "onehot",
        keep_original: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        enc_method = method
        if method == "categorical_encode":
            enc_method = kwargs.get("strategy", "onehot")
        if enc_method not in ("onehot", "label"):
            raise ValueError("method/strategy must be 'onehot' or 'label'")

        # Columns to encode
        if column:
            cols_to_encode = [column]
        else:
            cols_to_encode = df.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(f"Encoding columns: {cols_to_encode} using {enc_method}")

        # Track encoding metadata for orchestrator
        encoding_metadata = {
            'method': enc_method,
            'encoded_columns': {}  # {original_col: {'new_columns': [...], 'removed': bool}}
        }

        for col in cols_to_encode:
            if enc_method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, dtype=np.uint8, drop=None)
                encoded_data = encoder.fit_transform(df[[col]])
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)

                if keep_original:
                    df = pd.concat([df, encoded_df], axis=1)
                    encoding_metadata['encoded_columns'][col] = {
                        'new_columns': encoded_cols,
                        'removed': False
                    }
                else:
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                    encoding_metadata['encoded_columns'][col] = {
                        'new_columns': encoded_cols,
                        'removed': True
                    }

                logger.info(f"Applied OneHotEncoder on column '{col}' -> created {len(encoded_cols)} columns")
            else:  # label encoding
                encoder = LabelEncoder()
                encoded_col_name = f"{col}_encoded" if keep_original else col
                df[encoded_col_name] = encoder.fit_transform(df[col])
                
                if keep_original:
                    encoding_metadata['encoded_columns'][col] = {
                        'new_columns': [encoded_col_name],
                        'removed': False
                    }
                else:
                    # Column already replaced in-place, no need to drop
                    encoding_metadata['encoded_columns'][col] = {
                        'new_columns': [col],  # Same column name, now numeric
                        'removed': False  # Not removed, just transformed in-place
                    }

                logger.info(f"Applied LabelEncoder on column '{col}' as '{encoded_col_name}'")

        # Fill NaNs in numeric columns (after encoding)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Encoding complete. Final DataFrame shape: {df.shape}")
        
        # Store metadata as attribute for orchestrator to access
        if return_metadata:
            return df, encoding_metadata
        return df


# import pandas as pd
# import numpy as np

# try:
#     from ..base_agent import BaseAgent
#     from ...utils.logger import get_logger
# except ImportError:
#     from agents.base_agent import BaseAgent
#     from utils.logger import get_logger

# logger = get_logger(__name__)


# class EncodingAgent(BaseAgent):
#     """Agent for encoding categorical columns using one-hot or label encoding."""

#     def __init__(self):
#         super().__init__("EncodingAgent")

#     def process(self, df: pd.DataFrame, column: str = None, method: str = "onehot", **kwargs) -> pd.DataFrame:
#         """Encode categorical columns using one-hot or label encoding."""
#         df = df.copy().reset_index(drop=True)

#         # Normalize method for compatibility with orchestrator
#         enc_method = method
#         if method == "categorical_encode":
#             enc_method = kwargs.get("strategy", "onehot")
#         if enc_method not in ("onehot", "label"):
#             raise ValueError("method/strategy must be 'onehot' or 'label'")

#         # Handle single-column encoding
#         if column:
#             if enc_method == "onehot":
#                 dummies = pd.get_dummies(df[column].astype("category"), prefix=column, dtype=np.uint8)
#                 df = pd.concat([df.drop(columns=[column]).reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
#                 df = df.loc[:, ~df.columns.duplicated()]  # remove accidental duplicate cols
#                 logger.info(f"Applied one-hot encoding on column '{column}'")
#             else:  # label encoding
#                 df[column] = df[column].astype("category").cat.codes
#                 logger.info(f"Applied label encoding on column '{column}'")

#         else:
#             # Auto-detect categorical columns
#             cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
#             logger.info(f"Detected {len(cat_cols)} categorical columns for encoding: {cat_cols}")

#             for col in cat_cols:
#                 if enc_method == "onehot":
#                     dummies = pd.get_dummies(df[col].astype("category"), prefix=col, dtype=np.uint8)
#                     df = pd.concat([df.drop(columns=[col]).reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
#                     df = df.loc[:, ~df.columns.duplicated()]
#                     logger.info(f"Applied one-hot encoding on column '{col}'")
#                 else:
#                     df[col] = df[col].astype("category").cat.codes
#                     logger.info(f"Applied label encoding on column '{col}'")

#         # --- Key fixes below ---

#         # Convert to numeric where possible (ensures compatibility with visualizer)
#         df = df.apply(pd.to_numeric, errors="ignore")

#         # Drop any leftover non-numeric columns
#         non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
#         if len(non_numeric_cols) > 0:
#             logger.warning(f"Dropping non-numeric columns after encoding: {list(non_numeric_cols)}")
#             df = df.drop(columns=non_numeric_cols)

#         # Fill NaN (some encodings may introduce them)
#         df = df.fillna(0)

#         # Remove duplicate columns just in case
#         df = df.loc[:, ~df.columns.duplicated()]
#         df.reset_index(drop=True, inplace=True)

#         logger.info(f"Encoding complete. Final DataFrame shape: {df.shape}")
#         return df
