import pytest
from unittest.mock import patch, Mock

@pytest.mark.unit
class TestDownloadRouter:
    
    def test_download_dataset_success(self, test_client):
        """Test successful dataset download"""
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/download/dataset/test.csv")
            
            assert response.status_code == 200
    
    def test_download_dataset_not_found(self, test_client):
        """Test dataset download when file doesn't exist"""
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/download/dataset/nonexistent.csv")
            
            assert response.status_code == 404
            assert "Processed dataset not found" in response.json()["detail"]
    
    def test_download_report_success(self, test_client):
        """Test successful report download"""
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/download/report/test.html")
            
            assert response.status_code == 200
    
    def test_download_report_not_found(self, test_client):
        """Test report download when file doesn't exist"""
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/download/report/nonexistent.html")
            
            assert response.status_code == 404
            assert "Report not found" in response.json()["detail"]
    
    def test_list_processed_files_empty(self, test_client):
        """Test listing processed files when directory is empty"""
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_dir.exists.return_value = False
            
            response = test_client.get("/api/download/list")
            
            assert response.status_code == 200
            data = response.json()
            assert data["files"] == []
            assert data["count"] == 0
    
    def test_list_processed_files_with_files(self, test_client):
        """Test listing processed files with existing files"""
        from unittest.mock import MagicMock
        
        # Mock file objects
        mock_file1 = MagicMock()
        mock_file1.name = "processed_data.csv"
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1024
        mock_file1.stat.return_value.st_mtime = 1234567890
        mock_file1.suffix = ".csv"
        
        mock_file2 = MagicMock()
        mock_file2.name = "report.html"
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2048
        mock_file2.stat.return_value.st_mtime = 1234567891
        mock_file2.suffix = ".html"
        
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_dir.exists.return_value = True
            mock_dir.iterdir.return_value = [mock_file1, mock_file2]
            
            response = test_client.get("/api/download/list")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["files"]) == 2
            assert data["count"] == 2
            
            # Check file types
            types = [f["type"] for f in data["files"]]
            assert "dataset" in types
            assert "report" in types