import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from io import BytesIO

# Import your app
try:
    from server.main import app
    from server.agents.orchestrator import PipelineOrchestrator
except ImportError:
    # Handle import errors gracefully for testing
    app = None
    PipelineOrchestrator = None

@pytest.fixture
def test_client():
    """FastAPI test client"""
    if app is None:
        pytest.skip("FastAPI app not available")
    return TestClient(app)    #fake http

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'age': [25, 30, 35, None, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    })

@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing"""
    return {
        "users": [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
            {"id": 3, "name": "Charlie", "active": True}
        ]
    }

@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)

@pytest.fixture
def temp_json_file(sample_json_data):
    """Temporary JSON file for testing"""
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json_data, f)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)

@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing"""
    if PipelineOrchestrator is None:
        return Mock()
    orchestrator = Mock(spec=PipelineOrchestrator)
    orchestrator.run_pipeline = AsyncMock()
    return orchestrator

@pytest.fixture
def mock_spark_session():
    """Mock Spark session"""
    spark = Mock()
    spark.read.csv.return_value = Mock()
    spark.createDataFrame.return_value = Mock()
    return spark

@pytest.fixture
def csv_file_upload(sample_csv_data):
    """File upload fixture for CSV"""
    csv_content = sample_csv_data.to_csv(index=False)
    return {"file": ("test.csv", BytesIO(csv_content.encode()), "text/csv")}

@pytest.fixture
def invalid_file_upload():
    """Invalid file upload fixture"""
    return {"file": ("test.txt", BytesIO(b"invalid content"), "text/plain")}