import pytest
from io import BytesIO
from unittest.mock import patch

@pytest.mark.unit
class TestUploadRouter:
    
    def test_upload_csv_success(self, test_client, csv_file_upload):
        """Test successful CSV upload"""
        with patch('server.router.upload.save_upload') as mock_save:
            mock_save.return_value = "uploads/test.csv"
            
            response = test_client.post("/api/upload/", files=csv_file_upload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test.csv"
            assert "path" in data
            assert data["message"] == "File uploaded successfully"
    
    def test_upload_invalid_extension(self, test_client, invalid_file_upload):
        """Test upload with invalid file extension"""
        response = test_client.post("/api/upload/", files=invalid_file_upload)
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_no_file(self, test_client):
        """Test upload without file"""
        files = {"file": ("", BytesIO(b""), "")}
        response = test_client.post("/api/upload/", files=files)
        
        assert response.status_code == 400
        assert "No file selected" in response.json()["detail"]
    
    def test_upload_file_too_large(self, test_client):
        """Test upload with file too large"""
        # Create a large file content (simulate 1.1GB)
        large_content = "x" * (1024 * 1024 * 10)  # 10MB for testing
        files = {"file": ("large.csv", BytesIO(large_content.encode()), "text/csv")}
        
        with patch('server.config.MAX_UPLOAD_MB', 5):  # Set limit to 5MB
            response = test_client.post("/api/upload/", files=files)
            
            assert response.status_code == 413
            assert "File too large" in response.json()["detail"]
    
    def test_list_uploaded_files_empty(self, test_client):
        """Test listing uploaded files when directory is empty"""
        with patch('server.router.upload.UPLOAD_DIR') as mock_dir:
            mock_dir.exists.return_value = False
            
            response = test_client.get("/api/upload/list")
            
            assert response.status_code == 200
            data = response.json()
            assert data["files"] == []
            assert data["count"] == 0
    
    def test_list_uploaded_files_with_files(self, test_client):
        """Test listing uploaded files with existing files"""
        from pathlib import Path
        from unittest.mock import MagicMock
        
        # Mock file objects
        mock_file1 = MagicMock()
        mock_file1.name = "test1.csv"
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1024
        mock_file1.stat.return_value.st_mtime = 1234567890
        mock_file1.suffix = ".csv"
        
        mock_file2 = MagicMock()
        mock_file2.name = "test2.json"
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2048
        mock_file2.stat.return_value.st_mtime = 1234567891
        mock_file2.suffix = ".json"
        
        with patch('server.router.upload.UPLOAD_DIR') as mock_dir:
            mock_dir.exists.return_value = True
            mock_dir.iterdir.return_value = [mock_file1, mock_file2]
            
            response = test_client.get("/api/upload/list")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["files"]) == 2
            assert data["count"] == 2
            
            # Check file details
            filenames = [f["filename"] for f in data["files"]]
            assert "test1.csv" in filenames
            assert "test2.json" in filenames