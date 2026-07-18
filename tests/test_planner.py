"""Tests for Planner module."""

import pytest
from unittest.mock import Mock, patch

from cryoemagent.core.planner import (
    Planner,
    Plan,
    PlannedAction,
    ActionType,
)
from cryoemagent.core.memory import Memory
from cryoemagent.config import LLMConfig


class TestPlannedAction:
    """Tests for PlannedAction class."""
    
    def test_creation(self):
        action = PlannedAction(
            action_type=ActionType.CREATE_JOB,
            parameters={"job_type": "import_movies"},
            reasoning="Starting workflow",
        )
        
        assert action.action_type == ActionType.CREATE_JOB
        assert action.parameters["job_type"] == "import_movies"
    
    def test_to_dict(self):
        action = PlannedAction(
            action_type=ActionType.QUEUE_JOB,
            parameters={"lane": "default"},
            reasoning="Queue for execution",
        )
        
        data = action.to_dict()
        
        assert data["action_type"] == "queue_job"
        assert data["parameters"]["lane"] == "default"
        assert data["reasoning"] == "Queue for execution"


class TestPlan:
    """Tests for Plan class."""
    
    def test_creation(self):
        actions = [
            PlannedAction(ActionType.CREATE_JOB, {"job_type": "import"}, "Create job"),
            PlannedAction(ActionType.QUEUE_JOB, {}, "Queue job"),
        ]
        
        plan = Plan(
            goal="Import movies",
            actions=actions,
            contingency="Retry on failure",
        )
        
        assert plan.goal == "Import movies"
        assert len(plan.actions) == 2
        assert plan.contingency == "Retry on failure"
    
    def test_to_dict(self):
        plan = Plan(
            goal="Test goal",
            actions=[PlannedAction(ActionType.FINISH, {"reason": "Done"}, "Complete")],
        )
        
        data = plan.to_dict()
        
        assert data["goal"] == "Test goal"
        assert len(data["actions"]) == 1


class TestPlanner:
    """Tests for Planner class."""
    
    def test_get_initial_plan(self):
        config = LLMConfig(api_key="test-key")
        planner = Planner(config)
        
        plan = planner.get_initial_plan(
            movies_path="/data/movies/*.mrc",
            params={"pixel_size": 1.05, "voltage": 300},
        )
        
        assert plan.goal is not None
        assert len(plan.actions) >= 1
        assert plan.actions[0].action_type == ActionType.CREATE_JOB
    
    def test_react_decide_fallback_on_empty_api_key(self):
        config = LLMConfig(api_key="")
        planner = Planner(config)
        result = planner.react_decide("state", "quality", "history")
        assert result["decision"] in ("CONTINUE", "ADJUST", "ESCALATE")

    def test_fallback_decision_returns_continue(self):
        config = LLMConfig(api_key="test-key")
        planner = Planner(config)
        result = planner._fallback_decision("simulated error")
        assert result["decision"] == "CONTINUE"
        assert "reasoning" in result


class TestActionType:
    """Tests for ActionType enum."""
    
    def test_values(self):
        assert ActionType.CREATE_JOB.value == "create_job"
        assert ActionType.QUEUE_JOB.value == "queue_job"
        assert ActionType.WAIT_JOB.value == "wait_job"
        assert ActionType.FINISH.value == "finish"
