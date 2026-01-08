import pytest
from io import BytesIO
from unittest.mock import patch, Mock

@pytest.mark.integration
class TestAPIIntegration:
    
    def test_health_endpoints(self, test_client):
        """Test health check endpoints"""
        # Basic health
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "AI Data Cleaner"
        
        # API health
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "endpoints" in data
    
    def test_upload_and_list_flow(self, test_client, csv_file_upload):
        """Test upload -> list files flow"""
        # Mock the save_upload function
        with patch('server.router.upload.save_upload') as mock_save:
            mock_save.return_value = "uploads/test.csv"
            
            # 1. Upload file
            upload_response = test_client.post("/api/upload/", files=csv_file_upload)
            assert upload_response.status_code == 200
            
            filename = upload_response.json()["filename"]
            assert filename == "test.csv"
            
            # 2. Mock file listing
            from unittest.mock import MagicMock
            mock_file = MagicMock()
            mock_file.name = filename
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_size = 1024
            mock_file.stat.return_value.st_mtime = 1234567890
            mock_file.suffix = ".csv"
            
            with patch('server.router.upload.UPLOAD_DIR') as mock_dir:
                mock_dir.exists.return_value = True
                mock_dir.iterdir.return_value = [mock_file]
                
                # 3. List files
                list_response = test_client.get("/api/upload/list")
                assert list_response.status_code == 200
                
                files_list = list_response.json()["files"]
                assert len(files_list) == 1
                assert files_list[0]["filename"] == filename
    
    def test_full_pipeline_mock(self, test_client, csv_file_upload):
        """Test complete upload -> process -> download flow (mocked)"""
        # 1. Upload file
        with patch('server.router.upload.save_upload') as mock_save:
            mock_save.return_value = "uploads/integration_test.csv"
            
            upload_response = test_client.post("/api/upload/", files=csv_file_upload)
            assert upload_response.status_code == 200
            filename = upload_response.json()["filename"]
        
        # 2. Start processing (mock)
        with patch('server.router.process.UPLOAD_DIR') as mock_upload_dir, \
             patch('server.router.process.PipelineOrchestrator') as mock_orch_class:
            
            # Mock file exists
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_upload_dir.__truediv__.return_value = mock_path
            
            # Mock orchestrator
            mock_orch = Mock()
            async def mock_pipeline():
                yield {"type": "progress", "message": "Starting", "progress": 10}
                yield {"type": "progress", "message": "Processing", "progress": 50}
                yield {
                    "type": "complete", 
                    "message": "Done", 
                    "progress": 100,
                    "output_file": "processed_test.csv"
                }
            
            mock_orch.run_pipeline.return_value = mock_pipeline()
            mock_orch_class.return_value = mock_orch
            
            process_response = test_client.get(f"/api/process/start/{filename}")
            assert process_response.status_code == 200
            assert "text/event-stream" in process_response.headers["content-type"]
        
        # 3. Download processed file (mock)
        with patch('server.router.download.PROCESSED_DIR') as mock_processed_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_processed_dir.__truediv__.return_value = mock_path
            
            download_response = test_client.get("/api/download/dataset/processed_test.csv")
            assert download_response.status_code == 200
    
    def test_error_handling_flow(self, test_client):
        """Test error handling across endpoints"""
        # Test upload with invalid file
        invalid_files = {"file": ("test.txt", BytesIO(b"content"), "text/plain")}
        response = test_client.post("/api/upload/", files=invalid_files)
        assert response.status_code == 400
        
        # Test process with non-existent file
        with patch('server.router.process.UPLOAD_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/process/start/nonexistent.csv")
            assert response.status_code == 404
        
        # Test download with non-existent file
        with patch('server.router.download.PROCESSED_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/download/dataset/nonexistent.csv")
            assert response.status_code == 404