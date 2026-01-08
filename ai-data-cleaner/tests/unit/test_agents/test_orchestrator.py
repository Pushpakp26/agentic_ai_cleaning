import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

try:
    from server.agents.orchestrator import PipelineOrchestrator
except ImportError:
    PipelineOrchestrator = None

@pytest.mark.unit
class TestPipelineOrchestrator:
    
    @pytest.fixture
    def orchestrator(self, temp_csv_file):
        if PipelineOrchestrator is None:
            pytest.skip("PipelineOrchestrator not available")
        return PipelineOrchestrator(temp_csv_file)
    
    def test_init(self, orchestrator, temp_csv_file):
        """Test orchestrator initialization"""
        assert orchestrator.input_file == temp_csv_file
        assert orchestrator.session_id is not None
        assert orchestrator.use_spark in [True, False]
        assert orchestrator.current_df is None
        assert orchestrator.original_df is None
        assert isinstance(orchestrator.applied_ops, list)
    
    @pytest.mark.asyncio
    async def test_load_data_pandas(self, orchestrator):
        """Test loading data with Pandas"""
        orchestrator.use_spark = False
        
        with patch('server.agents.orchestrator.read_pandas') as mock_read:
            mock_df = pd.DataFrame({'col1': [1, 2, 3]})
            mock_read.return_value = mock_df
            
            await orchestrator._load_data()
            
            assert orchestrator.current_df is not None
            assert orchestrator.original_df is not None
            mock_read.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_data_spark(self, orchestrator, mock_spark_session):
        """Test loading data with Spark"""
        orchestrator.use_spark = True
        
        with patch('server.agents.orchestrator.get_spark', return_value=mock_spark_session):
            mock_df = Mock()
            mock_df.count.return_value = 100
            mock_df.columns = ['col1', 'col2']
            mock_df.cache.return_value = mock_df
            mock_df.limit.return_value.toPandas.return_value = pd.DataFrame({'col1': [1, 2]})
            mock_spark_session.read.option.return_value.option.return_value.csv.return_value = mock_df
            
            await orchestrator._load_data()
            
            assert orchestrator.current_df is not None
    
    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, orchestrator):
        """Test successful pipeline execution"""
        orchestrator.use_spark = False
        
        # Mock all pipeline steps
        orchestrator._load_data = AsyncMock()
        orchestrator._inspect_data = AsyncMock()
        orchestrator._apply_preprocessing_agents = AsyncMock()
        orchestrator._generate_visualizations = AsyncMock()
        orchestrator._create_summary_report = AsyncMock()
        orchestrator._generate_comparison_report = AsyncMock()
        orchestrator._save_final_dataset = AsyncMock(return_value=Path('test.csv'))
        
        messages = []
        async for message in orchestrator.run_pipeline():
            messages.append(message)
        
        # Check progress messages
        assert len(messages) > 0
        assert any(msg['type'] == 'progress' for msg in messages)
        assert any(msg['type'] == 'complete' for msg in messages)
        
        # Check completion message
        completion_msg = next(msg for msg in messages if msg['type'] == 'complete')
        assert completion_msg['progress'] == 100
        assert 'output_file' in completion_msg
    
    @pytest.mark.asyncio
    async def test_run_pipeline_error(self, orchestrator):
        """Test pipeline error handling"""
        orchestrator._load_data = AsyncMock(side_effect=Exception("Test error"))
        
        messages = []
        async for message in orchestrator.run_pipeline():
            messages.append(message)
        
        # Should have error message
        assert any(msg['type'] == 'error' for msg in messages)
        error_msg = next(msg for msg in messages if msg['type'] == 'error')
        assert 'Test error' in error_msg['message']
    
    def test_get_agent_for_suggestion(self, orchestrator):
        """Test agent selection for suggestions"""
        # Test pandas agents
        agent = orchestrator._get_agent_for_suggestion("fill_missing", use_spark=False)
        assert agent is not None
        
        # Test spark agents (if available)
        agent = orchestrator._get_agent_for_suggestion("fill_missing", use_spark=True)
        # May be None if spark agents not initialized
        
        # Test unknown suggestion
        agent = orchestrator._get_agent_for_suggestion("unknown_operation", use_spark=False)
        assert agent is None