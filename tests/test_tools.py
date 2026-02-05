"""Tests for CryoSPARC tools module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from cryoemagent.tools.cryosparc import CryoSPARCTools, JOB_TYPE_MAP
from cryoemagent.tools.base import ToolResult
from cryoemagent.config import CryoSPARCConfig


class TestJobTypeMap:
    """Tests for job type mapping."""
    
    def test_common_job_types(self):
        assert JOB_TYPE_MAP["import_movies"] == "import_movies"
        assert JOB_TYPE_MAP["patch_motion_correction"] == "patch_motion_correction_multi"
        assert JOB_TYPE_MAP["class_2d"] == "class_2D_new"
        assert JOB_TYPE_MAP["homo_refine"] == "homo_refine_new"


class TestCryoSPARCTools:
    """Tests for CryoSPARCTools class."""
    
    @pytest.fixture
    def mock_config(self):
        return CryoSPARCConfig(
            url="http://localhost:39000",
            email="test@example.com",
            password="password",
        )
    
    def test_initialization(self, mock_config):
        tools = CryoSPARCTools(mock_config)
        
        assert tools.config == mock_config
        assert tools._client is None
    
    @patch("cryoemagent.tools.cryosparc.CryoSPARC")
    def test_test_connection_success(self, mock_cs_class, mock_config):
        mock_client = Mock()
        mock_client.test_connection.return_value = True
        mock_cs_class.return_value = mock_client
        
        tools = CryoSPARCTools(mock_config)
        result = tools.test_connection()
        
        assert result.is_success()
        assert result.data["connected"] is True
    
    @patch("cryoemagent.tools.cryosparc.CryoSPARC")
    def test_test_connection_failure(self, mock_cs_class, mock_config):
        mock_cs_class.side_effect = Exception("Connection failed")
        
        tools = CryoSPARCTools(mock_config)
        result = tools.test_connection()
        
        assert not result.is_success()
        assert "Connection failed" in result.error
    
    @patch("cryoemagent.tools.cryosparc.CryoSPARC")
    def test_create_job_success(self, mock_cs_class, mock_config):
        mock_job = Mock()
        mock_job.uid = "J42"
        mock_job.status = "building"
        
        mock_client = Mock()
        mock_client.create_job.return_value = mock_job
        mock_cs_class.return_value = mock_client
        
        tools = CryoSPARCTools(mock_config)
        result = tools.create_job(
            project_uid="P1",
            workspace_uid="W1",
            job_type="import_movies",
        )
        
        assert result.is_success()
        assert result.data["job_uid"] == "J42"
    
    @patch("cryoemagent.tools.cryosparc.CryoSPARC")
    def test_queue_job_success(self, mock_cs_class, mock_config):
        mock_job = Mock()
        mock_job.status = "queued"
        
        mock_client = Mock()
        mock_client.find_job.return_value = mock_job
        mock_cs_class.return_value = mock_client
        
        tools = CryoSPARCTools(mock_config)
        result = tools.queue_job(
            job_uid="J1",
            project_uid="P1",
            lane="worker",
        )
        
        assert result.is_success()
        mock_job.queue.assert_called_once()


class TestToolResult:
    """Tests for ToolResult class."""
    
    def test_success(self):
        result = ToolResult.success({"key": "value"}, "Success message")
        
        assert result.is_success()
        assert result.data["key"] == "value"
        assert result.message == "Success message"
    
    def test_failure(self):
        result = ToolResult.failure("Error occurred")
        
        assert not result.is_success()
        assert result.error == "Error occurred"
