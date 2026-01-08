import pytest
from unittest.mock import patch, AsyncMock, Mock

@pytest.mark.unit
class TestProcessRouter:
    
    def test_start_processing_file_not_found(self, test_client):
        """Test processing with non-existent file"""
        with patch('server.router.process.UPLOAD_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.get("/api/process/start/nonexistent.csv")
            
            assert response.status_code == 404
            assert "File not found" in response.json()["detail"]
    
    @patch('server.router.process.PipelineOrchestrator')
    def test_start_processing_success(self, mock_orchestrator_class, test_client):
        """Test successful processing start"""
        # Mock file exists
        with patch('server.router.process.UPLOAD_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_path
            
            # Mock orchestrator
            mock_orchestrator = Mock()
            
            async def mock_pipeline():
                yield {"type": "progress", "message": "Starting", "progress": 0}
                yield {"type": "complete", "message": "Done", "progress": 100}
            
            mock_orchestrator.run_pipeline.return_value = mock_pipeline()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            response = test_client.get("/api/process/start/test.csv")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_start_processing_post_method(self, test_client):
        """Test POST method for processing"""
        with patch('server.router.process.UPLOAD_DIR') as mock_dir:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_path
            
            response = test_client.post("/api/process/start/test.csv")
            
            # Should behave same as GET method
            assert response.status_code == 404