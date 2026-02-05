"""Tests for the main CryoEMAgent."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from cryoemagent.core.agent import CryoEMAgent, AgentResult, AgentState
from cryoemagent.config import Config


class TestAgentResult:
    """Tests for AgentResult class."""
    
    def test_success_result(self):
        result = AgentResult(
            success=True,
            project_uid="P1",
            workspace_uid="W1",
            final_job_uid="J10",
            resolution=3.2,
            particle_count=50000,
            total_jobs=5,
            execution_time_seconds=3600.0,
        )
        
        assert result.success is True
        assert result.resolution == 3.2
        assert result.particle_count == 50000
    
    def test_failure_result(self):
        result = AgentResult(
            success=False,
            project_uid="P1",
            workspace_uid="W1",
            error="Connection failed",
        )
        
        assert result.success is False
        assert result.error == "Connection failed"
    
    def test_to_dict(self):
        result = AgentResult(
            success=True,
            project_uid="P1",
            workspace_uid="W1",
            total_jobs=3,
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["project_uid"] == "P1"
        assert data["total_jobs"] == 3


class TestAgentState:
    """Tests for AgentState enum."""
    
    def test_states(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.PLANNING.value == "planning"
        assert AgentState.EXECUTING.value == "executing"
        assert AgentState.COMPLETED.value == "completed"


class TestCryoEMAgent:
    """Tests for CryoEMAgent class."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)
        config.cryosparc = Mock()
        config.cryosparc.url = "http://localhost:39000"
        config.cryosparc.email = "test@example.com"
        config.cryosparc.password = "password"
        config.cryosparc.validate.return_value = True
        config.llm = Mock()
        config.llm.api_key = "test-key"
        config.llm.model = "gpt-4"
        config.llm.temperature = 0.1
        config.llm.max_tokens = 4096
        config.llm.validate.return_value = True
        config.processing = Mock()
        config.processing.pixel_size = 1.05
        config.processing.voltage = 300
        config.processing.total_dose = 50.0
        config.processing.spherical_aberration = 2.7
        config.processing.amplitude_contrast = 0.1
        config.processing.gpcr = Mock()
        config.processing.gpcr.resolution_target = 3.5
        config.max_retries = 3
        config.validate.return_value = (True, [])
        return config
    
    @patch("cryoemagent.core.agent.CryoSPARCTools")
    @patch("cryoemagent.core.agent.Planner")
    @patch("cryoemagent.core.agent.QualityAssessment")
    def test_initialization(self, mock_quality, mock_planner, mock_tools, mock_config):
        agent = CryoEMAgent(mock_config)
        
        assert agent.state == AgentState.IDLE
        assert agent.memory is not None
    
    @patch("cryoemagent.core.agent.CryoSPARCTools")
    @patch("cryoemagent.core.agent.Planner")
    @patch("cryoemagent.core.agent.QualityAssessment")
    def test_get_status(self, mock_quality, mock_planner, mock_tools, mock_config):
        agent = CryoEMAgent(mock_config)
        
        status = agent.get_status()
        
        assert status["state"] == "idle"
        assert status["jobs_completed"] == 0


class TestAgentIntegration:
    """Integration tests for agent workflow."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)
        config.cryosparc = Mock()
        config.cryosparc.url = "http://localhost:39000"
        config.cryosparc.email = "test@example.com"
        config.cryosparc.password = "password"
        config.cryosparc.validate.return_value = True
        config.llm = Mock()
        config.llm.api_key = "test-key"
        config.llm.model = "gpt-4"
        config.llm.temperature = 0.1
        config.llm.max_tokens = 4096
        config.llm.validate.return_value = True
        config.processing = Mock()
        config.processing.pixel_size = 1.05
        config.processing.voltage = 300
        config.processing.total_dose = 50.0
        config.processing.spherical_aberration = 2.7
        config.processing.amplitude_contrast = 0.1
        config.processing.gpcr = Mock()
        config.processing.gpcr.resolution_target = 3.5
        config.max_retries = 3
        config.polling_interval = 30
        config.validate.return_value = (True, [])
        return config
    
    @patch("cryoemagent.core.agent.CryoSPARCTools")
    @patch("cryoemagent.core.agent.Planner")
    @patch("cryoemagent.core.agent.QualityAssessment")
    def test_run_connection_failure(self, mock_quality, mock_planner, mock_tools_class, mock_config):
        from cryoemagent.tools.base import ToolResult
        
        mock_tools = Mock()
        mock_tools.test_connection.return_value = ToolResult.failure("Connection failed")
        mock_tools_class.return_value = mock_tools
        
        agent = CryoEMAgent(mock_config)
        result = agent.run(
            project_uid="P1",
            workspace_uid="W1",
            movies_path="/data/*.mrc",
        )
        
        assert result.success is False
        assert "Connection failed" in result.error
