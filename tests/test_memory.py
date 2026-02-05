"""Tests for Memory module."""

import pytest
from datetime import datetime

from cryoemagent.core.memory import (
    Memory,
    SemanticMemory,
    EpisodicMemory,
    ProcessingState,
    JobRecord,
    JobStatus,
)


class TestSemanticMemory:
    """Tests for SemanticMemory class."""
    
    def test_initialization(self):
        memory = SemanticMemory()
        
        assert memory.objective == ""
        assert len(memory.available_tools) > 0
        assert "import_movies" in memory.available_tools
        assert "homo_refine" in memory.available_tools
    
    def test_domain_knowledge(self):
        memory = SemanticMemory()
        
        assert "gpcr_characteristics" in memory.domain_knowledge
        assert memory.domain_knowledge["gpcr_characteristics"]["typical_box_size"] == 256
        assert memory.domain_knowledge["gpcr_characteristics"]["expected_symmetry"] == "C1"
    
    def test_get_next_stage(self):
        memory = SemanticMemory()
        
        assert memory.get_next_stage("import") == "motion_correction"
        assert memory.get_next_stage("motion_correction") == "ctf_estimation"
        assert memory.get_next_stage("refinement") is None
    
    def test_to_prompt_context(self):
        memory = SemanticMemory()
        memory.objective = "Determine GPCR structure"
        
        context = memory.to_prompt_context()
        
        assert "OBJECTIVE: Determine GPCR structure" in context
        assert "AVAILABLE TOOLS:" in context
        assert "GPCR DOMAIN KNOWLEDGE:" in context


class TestEpisodicMemory:
    """Tests for EpisodicMemory class."""
    
    def test_initialization(self):
        memory = EpisodicMemory()
        
        assert memory.state is None
        assert len(memory.job_history) == 0
        assert len(memory.observations) == 0
    
    def test_add_observation(self):
        memory = EpisodicMemory()
        memory.add_observation("Test observation", {"key": "value"})
        
        assert len(memory.observations) == 1
        assert memory.observations[0]["observation"] == "Test observation"
        assert memory.observations[0]["data"]["key"] == "value"
    
    def test_add_decision(self):
        memory = EpisodicMemory()
        memory.add_decision("Execute import", "Starting workflow")
        
        assert len(memory.decisions) == 1
        assert memory.decisions[0]["decision"] == "Execute import"
        assert memory.decisions[0]["reasoning"] == "Starting workflow"
    
    def test_add_job(self):
        memory = EpisodicMemory()
        job = JobRecord(
            job_uid="J1",
            job_type="import_movies",
            status=JobStatus.COMPLETED,
            started_at=datetime.now(),
        )
        memory.add_job(job)
        
        assert len(memory.job_history) == 1
        assert memory.job_history[0].job_uid == "J1"
    
    def test_get_last_completed_job(self):
        memory = EpisodicMemory()
        
        job1 = JobRecord(
            job_uid="J1",
            job_type="import_movies",
            status=JobStatus.COMPLETED,
            started_at=datetime.now(),
        )
        job2 = JobRecord(
            job_uid="J2",
            job_type="motion_correction",
            status=JobStatus.COMPLETED,
            started_at=datetime.now(),
        )
        memory.add_job(job1)
        memory.add_job(job2)
        
        last_job = memory.get_last_completed_job()
        assert last_job.job_uid == "J2"
        
        import_job = memory.get_last_completed_job("import_movies")
        assert import_job.job_uid == "J1"


class TestProcessingState:
    """Tests for ProcessingState class."""
    
    def test_initialization(self):
        state = ProcessingState(project_uid="P1", workspace_uid="W1")
        
        assert state.project_uid == "P1"
        assert state.workspace_uid == "W1"
        assert state.current_stage == "initialization"
        assert not state.movies_imported
    
    def test_to_dict(self):
        state = ProcessingState(project_uid="P1", workspace_uid="W1")
        state.movies_imported = True
        state.total_movies = 100
        
        data = state.to_dict()
        
        assert data["project_uid"] == "P1"
        assert data["pipeline_status"]["movies_imported"] is True
        assert data["counts"]["movies"] == 100


class TestMemory:
    """Tests for combined Memory class."""
    
    def test_initialization(self):
        memory = Memory()
        
        assert memory.semantic is not None
        assert memory.episodic is not None
    
    def test_initialize_session(self):
        memory = Memory()
        memory.initialize("P1", "W1", "Test objective")
        
        assert memory.semantic.objective == "Test objective"
        assert memory.episodic.state is not None
        assert memory.episodic.state.project_uid == "P1"
    
    def test_get_full_context(self):
        memory = Memory()
        memory.initialize("P1", "W1", "Determine structure")
        
        context = memory.get_full_context()
        
        assert "OBJECTIVE: Determine structure" in context
        assert "CURRENT SESSION:" in context
    
    def test_to_dict(self):
        memory = Memory()
        memory.initialize("P1", "W1", "Test")
        
        data = memory.to_dict()
        
        assert "semantic" in data
        assert "episodic" in data
        assert data["semantic"]["objective"] == "Test"
