import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock

try:
    from server.utils.file_handler import read_pandas, write_pandas, detect_file_kind, save_upload
except ImportError:
    read_pandas = write_pandas = detect_file_kind = save_upload = None

@pytest.mark.unit
class TestFileHandler:
    
    def test_detect_file_kind_csv(self):
        """Test CSV file detection"""
        if detect_file_kind is None:
            pytest.skip("detect_file_kind not available")
        
        assert detect_file_kind(Path("test.csv")) == "csv"
        assert detect_file_kind(Path("test.CSV")) == "csv"
    
    def test_detect_file_kind_json(self):
        """Test JSON file detection"""
        if detect_file_kind is None:
            pytest.skip("detect_file_kind not available")
        
        assert detect_file_kind(Path("test.json")) == "json"
        assert detect_file_kind(Path("TEST.JSON")) == "json"
    
    def test_detect_file_kind_parquet(self):
        """Test Parquet file detection"""
        if detect_file_kind is None:
            pytest.skip("detect_file_kind not available")
        
        assert detect_file_kind(Path("test.parquet")) == "parquet"
        assert detect_file_kind(Path("test.PARQUET")) == "parquet"
    
    def test_detect_file_kind_unknown(self):
        """Test unknown file type detection"""
        if detect_file_kind is None:
            pytest.skip("detect_file_kind not available")
        
        # Should raise ValueError for unknown extensions
        with pytest.raises(ValueError, match="Unsupported extension"):
            detect_file_kind(Path("test.unknown"))
    
    def test_read_pandas_csv(self, temp_csv_file):
        """Test reading CSV with pandas"""
        if read_pandas is None:
            pytest.skip("read_pandas not available")
        
        df = read_pandas(temp_csv_file, "csv")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'id' in df.columns
    
    def test_read_pandas_json(self, temp_json_file):
        """Test reading JSON with pandas"""
        if read_pandas is None:
            pytest.skip("read_pandas not available")
        
        # Mock pandas read_json for JSON files
        with patch('pandas.read_json') as mock_read:
            mock_df = pd.DataFrame({'col1': [1, 2, 3]})
            mock_read.return_value = mock_df
            
            df = read_pandas(temp_json_file, "json")
            
            assert isinstance(df, pd.DataFrame)
            mock_read.assert_called_once()
    
    def test_write_pandas_csv(self, sample_csv_data, tmp_path):
        """Test writing CSV with pandas"""
        if write_pandas is None:
            pytest.skip("write_pandas not available")
        
        output_file = "test_output.csv"
        
        with patch('server.utils.file_handler.PROCESSED_DIR', tmp_path):
            result_path = write_pandas(sample_csv_data, output_file, "csv")
            
            assert result_path.exists()
            assert result_path.suffix == ".csv"
            
            # Verify content
            df_read = pd.read_csv(result_path)
            assert len(df_read) == len(sample_csv_data)
    
    def test_write_pandas_json(self, sample_csv_data, tmp_path):
        """Test writing JSON with pandas"""
        if write_pandas is None:
            pytest.skip("write_pandas not available")
        
        output_file = "test_output.json"
        
        with patch('server.utils.file_handler.PROCESSED_DIR', tmp_path):
            result_path = write_pandas(sample_csv_data, output_file, "json")
            
            assert result_path.exists()
            assert result_path.suffix == ".json"
    
    def test_save_upload(self, tmp_path):
        """Test saving uploaded file"""
        if save_upload is None:
            pytest.skip("save_upload not available")
        
        content = b"test,data\n1,2\n3,4"
        filename = "test_upload.csv"
        
        with patch('server.utils.file_handler.UPLOAD_DIR', tmp_path):
            result_path = save_upload(content, filename)
            
            assert result_path.exists()
            assert result_path.name == filename
            assert result_path.read_bytes() == content