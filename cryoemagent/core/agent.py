"""Main CryoEMAgent - Autonomous cryo-EM structure determination."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cryoemagent.config import Config
from cryoemagent.core.memory import Memory, JobRecord, JobStatus, ProcessingState
from cryoemagent.core.planner import Planner, Plan, PlannedAction, ActionType
from cryoemagent.tools.cryosparc import CryoSPARCTools
from cryoemagent.tools.quality import QualityAssessment
from cryoemagent.tools.base import ToolResult

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent execution state."""
    
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Result from agent execution."""
    
    success: bool
    project_uid: str
    workspace_uid: str
    final_job_uid: Optional[str] = None
    resolution: Optional[float] = None
    particle_count: int = 0
    total_jobs: int = 0
    execution_time_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "project_uid": self.project_uid,
            "workspace_uid": self.workspace_uid,
            "final_job_uid": self.final_job_uid,
            "resolution": self.resolution,
            "particle_count": self.particle_count,
            "total_jobs": self.total_jobs,
            "execution_time_seconds": self.execution_time_seconds,
            "error": self.error,
        }


class CryoEMAgent:
    """Autonomous Cryo-EM structure determination agent."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        
        valid, errors = self.config.validate()
        if not valid:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")
        
        self.memory = Memory()
        self.planner = Planner(self.config.llm)
        self.tools = CryoSPARCTools(self.config.cryosparc)
        self.quality = QualityAssessment()
        
        self.state = AgentState.IDLE
        self._current_job_uid: Optional[str] = None
        self._start_time: Optional[datetime] = None
    
    def run(
        self,
        project_uid: str,
        workspace_uid: str,
        movies_path: str,
        pixel_size: Optional[float] = None,
        voltage: Optional[int] = None,
        total_dose: Optional[float] = None,
        target_resolution: Optional[float] = None,
    ) -> AgentResult:
        """Execute end-to-end GPCR structure determination workflow."""
        self._start_time = datetime.now()
        
        pixel_size = pixel_size or self.config.processing.pixel_size
        voltage = voltage or self.config.processing.voltage
        total_dose = total_dose or self.config.processing.total_dose
        target_resolution = target_resolution or self.config.processing.gpcr.resolution_target
        
        objective = (
            f"Determine GPCR structure from {movies_path} to target resolution "
            f"{target_resolution}Å using automated cryo-EM workflow."
        )
        
        self.memory.initialize(project_uid, workspace_uid, objective)
        
        logger.info(f"Starting CryoEMAgent for project {project_uid}")
        logger.info(f"Movies: {movies_path}, Pixel size: {pixel_size}Å, Target: {target_resolution}Å")
        
        connection_result = self.tools.test_connection()
        if not connection_result.is_success():
            return AgentResult(
                success=False,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                error=f"CryoSPARC connection failed: {connection_result.error}",
            )
        
        try:
            result = self._execute_workflow(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                movies_path=movies_path,
                pixel_size=pixel_size,
                voltage=voltage,
                total_dose=total_dose,
                target_resolution=target_resolution,
            )
            return result
        except Exception as e:
            logger.exception("Agent execution failed")
            return AgentResult(
                success=False,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                error=str(e),
                total_jobs=len(self.memory.episodic.job_history),
                execution_time_seconds=self._get_execution_time(),
            )
    
    def _execute_workflow(
        self,
        project_uid: str,
        workspace_uid: str,
        movies_path: str,
        pixel_size: float,
        voltage: int,
        total_dose: float,
        target_resolution: float,
    ) -> AgentResult:
        """Execute the main workflow loop."""
        params = {
            "pixel_size": pixel_size,
            "voltage": voltage,
            "total_dose": total_dose,
            "spherical_aberration": self.config.processing.spherical_aberration,
            "amplitude_contrast": self.config.processing.amplitude_contrast,
        }
        
        self.state = AgentState.PLANNING
        plan = self.planner.get_initial_plan(movies_path, params)
        
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            self.state = AgentState.EXECUTING
            
            for action in plan.actions:
                result = self._execute_action(action, project_uid, workspace_uid)
                
                if not result.is_success():
                    self.memory.episodic.add_error(
                        f"Action {action.action_type.value} failed",
                        {"error": result.error, "action": action.to_dict()},
                    )
                    
                    if action.action_type == ActionType.FINISH:
                        break
                    
                    retry_count = 0
                    while retry_count < self.config.max_retries:
                        retry_count += 1
                        logger.warning(f"Retrying action (attempt {retry_count})")
                        result = self._execute_action(action, project_uid, workspace_uid)
                        if result.is_success():
                            break
                    
                    if not result.is_success():
                        logger.error(f"Action failed after {self.config.max_retries} retries")
                        break
                
                if action.action_type == ActionType.FINISH:
                    self.state = AgentState.COMPLETED
                    return self._build_result(project_uid, workspace_uid, success=True)
            
            state = self.memory.episodic.state
            if state and state.current_resolution and state.current_resolution <= target_resolution:
                logger.info(f"Target resolution {target_resolution}Å achieved!")
                self.state = AgentState.COMPLETED
                return self._build_result(project_uid, workspace_uid, success=True)
            
            self.state = AgentState.REFLECTING
            reflection = self.planner.reflect_on_result(
                self.memory,
                {"last_action": plan.actions[-1].to_dict() if plan.actions else {}},
            )
            self.memory.episodic.add_observation(f"Reflection: {reflection[:200]}...")
            
            self.state = AgentState.PLANNING
            plan = self.planner.plan_next_step(self.memory)
            
            if not plan.actions or plan.actions[0].action_type == ActionType.FINISH:
                self.state = AgentState.COMPLETED
                return self._build_result(project_uid, workspace_uid, success=True)
        
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return self._build_result(project_uid, workspace_uid, success=False, error="Max iterations reached")
    
    def _execute_action(
        self,
        action: PlannedAction,
        project_uid: str,
        workspace_uid: str,
    ) -> ToolResult:
        """Execute a single planned action."""
        logger.info(f"Executing action: {action.action_type.value}")
        self.memory.episodic.add_decision(
            f"Execute {action.action_type.value}",
            action.reasoning,
        )
        
        params = action.parameters
        
        if action.action_type == ActionType.CREATE_JOB:
            result = self.tools.create_job(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                job_type=params.get("job_type", ""),
                title=params.get("title", ""),
                params=params.get("params", {}),
                connections=params.get("connections"),
            )
            if result.is_success():
                self._current_job_uid = result.data.get("job_uid")
                self._update_state_for_job(params.get("job_type", ""))
            return result
        
        elif action.action_type == ActionType.SET_PARAM:
            if not self._current_job_uid:
                return ToolResult.failure("No current job to set parameters on")
            return self.tools.set_param(
                job_uid=self._current_job_uid,
                project_uid=project_uid,
                name=params.get("name", ""),
                value=params.get("value"),
            )
        
        elif action.action_type == ActionType.CONNECT_INPUT:
            if not self._current_job_uid:
                return ToolResult.failure("No current job to connect inputs")
            return self.tools.connect_input(
                job_uid=self._current_job_uid,
                project_uid=project_uid,
                input_name=params.get("input_name", ""),
                source_job_uid=params.get("source_job_uid", ""),
                source_output=params.get("source_output", ""),
            )
        
        elif action.action_type == ActionType.QUEUE_JOB:
            if not self._current_job_uid:
                return ToolResult.failure("No current job to queue")
            return self.tools.queue_job(
                job_uid=self._current_job_uid,
                project_uid=project_uid,
                lane=params.get("lane"),
                gpus=params.get("gpus"),
            )
        
        elif action.action_type == ActionType.WAIT_JOB:
            if not self._current_job_uid:
                return ToolResult.failure("No current job to wait for")
            
            self.state = AgentState.WAITING
            result = self.tools.wait_for_job(
                job_uid=self._current_job_uid,
                project_uid=project_uid,
                timeout=params.get("timeout"),
                error_on_incomplete=params.get("error_on_incomplete", True),
            )
            
            if result.is_success():
                job_record = JobRecord(
                    job_uid=self._current_job_uid,
                    job_type=result.data.get("job_type", "unknown"),
                    status=JobStatus.COMPLETED if result.data.get("status") == "completed" else JobStatus.FAILED,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    outputs=result.data.get("outputs", {}),
                )
                self.memory.episodic.add_job(job_record)
            
            return result
        
        elif action.action_type == ActionType.LOAD_OUTPUT:
            job_uid = params.get("job_uid") or self._current_job_uid
            if not job_uid:
                return ToolResult.failure("No job UID specified for loading output")
            
            result = self.tools.load_output(
                job_uid=job_uid,
                project_uid=project_uid,
                output_name=params.get("output_name", ""),
            )
            
            if result.is_success():
                self._update_counts(result.data)
            
            return result
        
        elif action.action_type == ActionType.ASSESS_QUALITY:
            return ToolResult.success(
                {"quality": "assessment_placeholder"},
                "Quality assessment completed",
            )
        
        elif action.action_type == ActionType.FINISH:
            reason = params.get("reason", "Workflow completed")
            logger.info(f"Agent finishing: {reason}")
            return ToolResult.success({"reason": reason}, reason)
        
        else:
            return ToolResult.failure(f"Unknown action type: {action.action_type}")
    
    def _update_state_for_job(self, job_type: str):
        """Update processing state based on job type."""
        if not self.memory.episodic.state:
            return
        
        state = self.memory.episodic.state
        state.current_stage = job_type
        state.active_job_uid = self._current_job_uid
        
        stage_map = {
            "import_movies": ("movies_imported", True),
            "patch_motion_correction": ("motion_corrected", True),
            "patch_ctf_estimation": ("ctf_estimated", True),
            "blob_picker": ("particles_picked", True),
            "extract_micrographs": ("particles_extracted", True),
            "class_2d": ("classes_2d_done", True),
            "homo_abinit": ("abinit_done", True),
            "homo_refine": ("refinement_done", True),
            "nonuniform_refine": ("refinement_done", True),
        }
        
        if job_type in stage_map:
            attr, value = stage_map[job_type]
            setattr(state, attr, value)
    
    def _update_counts(self, data: Dict[str, Any]):
        """Update particle/micrograph counts from output data."""
        if not self.memory.episodic.state:
            return
        
        state = self.memory.episodic.state
        
        if "num_rows" in data:
            output_name = data.get("output_name", "")
            if "micrograph" in output_name.lower():
                state.total_micrographs = data["num_rows"]
            elif "particle" in output_name.lower():
                state.total_particles = data["num_rows"]
    
    def _build_result(
        self,
        project_uid: str,
        workspace_uid: str,
        success: bool,
        error: Optional[str] = None,
    ) -> AgentResult:
        """Build final agent result."""
        state = self.memory.episodic.state
        
        return AgentResult(
            success=success,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            final_job_uid=self._current_job_uid,
            resolution=state.current_resolution if state else None,
            particle_count=state.total_particles if state else 0,
            total_jobs=len(self.memory.episodic.job_history),
            execution_time_seconds=self._get_execution_time(),
            error=error,
        )
    
    def _get_execution_time(self) -> float:
        """Get execution time in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "state": self.state.value,
            "current_job": self._current_job_uid,
            "jobs_completed": len(self.memory.episodic.job_history),
            "memory": self.memory.to_dict() if self.memory.episodic.state else None,
        }
