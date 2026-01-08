# AI Data Cleaner - Test Suite

## Overview
Comprehensive test suite for the AI Data Cleaner project with unit, integration, and end-to-end tests.

## Test Structure
```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_agents/       # Agent-specific tests
│   ├── test_router/       # API router tests
│   └── test_utils/        # Utility function tests
├── integration/           # Integration tests for API endpoints
├── e2e/                  # End-to-end workflow tests
├── fixtures/             # Test data and mock responses
└── performance/          # Performance and load tests
```

## Running Tests

### Install Test Dependencies
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
pytest
# or
python run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -m unit
python run_tests.py --unit

# Integration tests only
pytest tests/integration/ -m integration
python run_tests.py --integration

# With coverage report
pytest --cov=server --cov-report=html
python run_tests.py --coverage
```

### Run Specific Test Files
```bash
pytest tests/unit/test_agents/test_orchestrator.py -v
pytest tests/unit/test_router/test_upload.py::TestUploadRouter::test_upload_csv_success
```

## Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.spark` - Tests requiring Spark

## Coverage Requirements
- Minimum coverage: 80%
- Generate HTML report: `pytest --cov=server --cov-report=html`
- View report: Open `htmlcov/index.html`

## CI/CD Integration
Tests run automatically on:
- Push to main/develop branches
- Pull requests to main
- Multiple Python versions (3.9, 3.10, 3.11)

## Writing New Tests
1. Follow naming convention: `test_*.py`
2. Use appropriate markers
3. Add fixtures to `conftest.py` for reusable test data
4. Mock external dependencies
5. Test both success and error cases