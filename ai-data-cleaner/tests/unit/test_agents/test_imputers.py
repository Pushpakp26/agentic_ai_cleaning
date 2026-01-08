import pytest
import pandas as pd
import numpy as np

try:
    from server.agents.imputers.mean_median_imputer import MeanMedianImputerAgent
except ImportError:
    MeanMedianImputerAgent = None

@pytest.mark.unit
class TestMeanMedianImputerAgent:
    
    @pytest.fixture
    def imputer(self):
        if MeanMedianImputerAgent is None:
            pytest.skip("MeanMedianImputerAgent not available")
        return MeanMedianImputerAgent()
    
    @pytest.fixture
    def df_with_missing(self):
        return pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'B'],
            'complete_col': [10, 20, 30, 40, 50]
        })
    
    def test_impute_numeric_mean(self, imputer, df_with_missing):
        """Test mean imputation for numeric columns"""
        result = imputer.process(df_with_missing, column='numeric_col', strategy='mean')
        
        assert not result['numeric_col'].isna().any()
        assert result['numeric_col'].iloc[2] == 3.0  # Mean of [1,2,4,5]
    
    def test_impute_numeric_median(self, imputer, df_with_missing):
        """Test median imputation for numeric columns"""
        result = imputer.process(df_with_missing, column='numeric_col', strategy='median')
        
        assert not result['numeric_col'].isna().any()
        assert result['numeric_col'].iloc[2] == 3.0  # Median of [1,2,4,5]
    
    def test_impute_categorical_mode(self, imputer, df_with_missing):
        """Test mode imputation for categorical columns"""
        result = imputer.process(df_with_missing, column='categorical_col', strategy='mode')
        
        assert not result['categorical_col'].isna().any()
        assert result['categorical_col'].iloc[2] in ['A', 'B']  # Most frequent value
    
    def test_auto_strategy_selection(self, imputer, df_with_missing):
        """Test automatic strategy selection"""
        result = imputer.process(df_with_missing, column='numeric_col', strategy='auto')
        
        assert not result['numeric_col'].isna().any()
    
    def test_no_missing_values(self, imputer, df_with_missing):
        """Test behavior when no missing values"""
        result = imputer.process(df_with_missing, column='complete_col', strategy='mean')
        
        pd.testing.assert_series_equal(result['complete_col'], df_with_missing['complete_col'])
    
    def test_all_columns_imputation(self, imputer, df_with_missing):
        """Test imputing all columns at once"""
        result = imputer.process(df_with_missing, strategy='auto')
        
        # Should have no missing values in any column
        assert not result.isna().any().any()
    
    def test_forward_fill_strategy(self, imputer):
        """Test forward fill strategy"""
        df = pd.DataFrame({
            'col': [1, np.nan, 3, np.nan, 5]
        })
        
        result = imputer.process(df, column='col', strategy='forward_fill')
        
        expected = [1, 1, 3, 3, 5]  # Forward filled
        assert result['col'].tolist() == expected
    
    def test_backward_fill_strategy(self, imputer):
        """Test backward fill strategy"""
        df = pd.DataFrame({
            'col': [1, np.nan, 3, np.nan, 5]
        })
        
        result = imputer.process(df, column='col', strategy='backward_fill')
        
        expected = [1, 3, 3, 5, 5]  # Backward filled
        assert result['col'].tolist() == expected