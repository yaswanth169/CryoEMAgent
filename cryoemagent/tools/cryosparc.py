"""CryoSPARC integration tools for job management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from cryosparc.tools import CryoSPARC
from cryosparc.controllers.job import JobController

from cryoemagent.tools.base import Tool, ToolResult
from cryoemagent.config import CryoSPARCConfig

logger = logging.getLogger(__name__)


JOB_TYPE_MAP = {
    "import_movies": "import_movies",
    "patch_motion_correction": "patch_motion_correction_multi",
    "patch_ctf_estimation": "patch_ctf_estimation_multi",
    "blob_picker": "blob_picker_gpu",
    "template_picker": "template_picker_gpu",
    "extract_micrographs": "extract_micrographs_multi",
    "class_2d": "class_2D_new",
    "select_2d": "select_2D",
    "homo_abinit": "homo_abinit",
    "hetero_refine": "hetero_refine",
    "homo_refine": "homo_refine_new",
    "nonuniform_refine": "nonuniform_refine_new",
    "local_refine": "local_refinement",
}


@dataclass
class JobInfo:
    """Information about a CryoSPARC job."""
    
    uid: str
    job_type: str
    status: str
    title: str = ""
    outputs: Dict[str, Any] = None
    
    def __post_init__(self):
        self.outputs = self.outputs or {}


class CryoSPARCTools:
    """CryoSPARC job management tools."""
    
    def __init__(self, config: CryoSPARCConfig):
        self.config = config
        self._client: Optional[CryoSPARC] = None
        self._current_job: Optional[JobController] = None
    
    @property
    def client(self) -> CryoSPARC:
        """Lazy initialization of CryoSPARC client."""
        if self._client is None:
            self._client = CryoSPARC(
                self.config.url,
                email=self.config.email,
                password=self.config.password,
            )
            logger.info(f"Connected to CryoSPARC at {self.config.url}")
        return self._client
    
    def test_connection(self) -> ToolResult:
        """Test connection to CryoSPARC."""
        try:
            if self.client.test_connection():
                return ToolResult.success(
                    {"connected": True},
                    f"Successfully connected to CryoSPARC at {self.config.url}"
                )
            return ToolResult.failure("Connection test returned False")
        except Exception as e:
            return ToolResult.failure(f"Connection failed: {str(e)}")
    
    def create_job(
        self,
        project_uid: str,
        workspace_uid: str,
        job_type: str,
        title: str = "",
        params: Optional[Dict[str, Any]] = None,
        connections: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> ToolResult:
        """Create a new CryoSPARC job."""
        try:
            cs_job_type = JOB_TYPE_MAP.get(job_type, job_type)
            
            job = self.client.create_job(
                project_uid,
                workspace_uid,
                cs_job_type,
                connections=connections or {},
                params=params or {},
                title=title or f"Auto: {job_type}",
            )
            
            self._current_job = job
            
            logger.info(f"Created job {job.uid} of type {cs_job_type}")
            
            return ToolResult.success(
                {
                    "job_uid": job.uid,
                    "job_type": cs_job_type,
                    "status": job.status,
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                },
                f"Created job {job.uid}"
            )
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return ToolResult.failure(f"Failed to create job: {str(e)}")
    
    def set_param(self, job_uid: str, project_uid: str, name: str, value: Any) -> ToolResult:
        """Set a parameter on a job."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            success = job.set_param(name, value)
            
            if success:
                return ToolResult.success(
                    {"job_uid": job_uid, "param": name, "value": value},
                    f"Set {name}={value} on {job_uid}"
                )
            return ToolResult.failure(f"Failed to set parameter {name}")
        except Exception as e:
            return ToolResult.failure(f"Failed to set parameter: {str(e)}")
    
    def connect_input(
        self,
        job_uid: str,
        project_uid: str,
        input_name: str,
        source_job_uid: str,
        source_output: str,
    ) -> ToolResult:
        """Connect a job input to another job's output."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            success = job.connect(input_name, source_job_uid, source_output)
            
            if success:
                return ToolResult.success(
                    {
                        "job_uid": job_uid,
                        "input": input_name,
                        "source": f"{source_job_uid}.{source_output}",
                    },
                    f"Connected {input_name} to {source_job_uid}.{source_output}"
                )
            return ToolResult.failure("Connection failed")
        except Exception as e:
            return ToolResult.failure(f"Failed to connect input: {str(e)}")
    
    def queue_job(
        self,
        job_uid: str,
        project_uid: str,
        lane: Optional[str] = None,
        gpus: Optional[List[int]] = None,
    ) -> ToolResult:
        """Queue a job for execution."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            job.queue(lane=lane, gpus=gpus or [])
            
            logger.info(f"Queued job {job_uid}")
            
            return ToolResult.success(
                {"job_uid": job_uid, "status": job.status, "lane": lane},
                f"Queued job {job_uid}"
            )
        except Exception as e:
            return ToolResult.failure(f"Failed to queue job: {str(e)}")
    
    def wait_for_job(
        self,
        job_uid: str,
        project_uid: str,
        timeout: Optional[int] = None,
        error_on_incomplete: bool = True,
    ) -> ToolResult:
        """Wait for a job to complete."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            
            logger.info(f"Waiting for job {job_uid} to complete...")
            
            status = job.wait_for_done(
                error_on_incomplete=error_on_incomplete,
                timeout=timeout,
            )
            
            job.refresh()
            
            outputs = {}
            for output_name, output_data in job.outputs.items():
                outputs[output_name] = {
                    "type": output_data.type if hasattr(output_data, 'type') else "unknown",
                    "count": output_data.num_items if hasattr(output_data, 'num_items') else 0,
                }
            
            return ToolResult.success(
                {
                    "job_uid": job_uid,
                    "status": status,
                    "outputs": outputs,
                },
                f"Job {job_uid} completed with status: {status}"
            )
        except AssertionError as e:
            return ToolResult.failure(f"Job did not complete successfully: {str(e)}")
        except Exception as e:
            return ToolResult.failure(f"Error waiting for job: {str(e)}")
    
    def load_output(
        self,
        job_uid: str,
        project_uid: str,
        output_name: str,
    ) -> ToolResult:
        """Load a job output dataset."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            dataset = job.load_output(output_name)
            
            summary = {
                "job_uid": job_uid,
                "output_name": output_name,
                "num_rows": len(dataset),
                "fields": list(dataset.fields()) if hasattr(dataset, 'fields') else [],
            }
            
            return ToolResult.success(summary, f"Loaded {len(dataset)} items from {output_name}")
        except Exception as e:
            return ToolResult.failure(f"Failed to load output: {str(e)}")
    
    def get_job_status(self, job_uid: str, project_uid: str) -> ToolResult:
        """Get current status of a job."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            job.refresh()
            
            return ToolResult.success(
                {
                    "job_uid": job_uid,
                    "status": job.status,
                    "type": job.type,
                    "title": job.title,
                },
                f"Job {job_uid} status: {job.status}"
            )
        except Exception as e:
            return ToolResult.failure(f"Failed to get job status: {str(e)}")
    
    def find_jobs(
        self,
        project_uid: str,
        workspace_uid: Optional[str] = None,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> ToolResult:
        """Find jobs matching criteria."""
        try:
            project = self.client.find_project(project_uid)
            
            kwargs = {}
            if workspace_uid:
                kwargs["workspace_uid"] = workspace_uid
            if job_type:
                kwargs["type"] = JOB_TYPE_MAP.get(job_type, job_type)
            if status:
                kwargs["status"] = status
            
            jobs = list(project.find_jobs(**kwargs))
            
            job_list = [
                {
                    "uid": job.uid,
                    "type": job.type,
                    "status": job.status,
                    "title": job.title,
                }
                for job in jobs
            ]
            
            return ToolResult.success(
                {"jobs": job_list, "count": len(job_list)},
                f"Found {len(job_list)} jobs"
            )
        except Exception as e:
            return ToolResult.failure(f"Failed to find jobs: {str(e)}")
    
    def create_workspace(self, project_uid: str, title: str) -> ToolResult:
        """Create a new workspace in a project."""
        try:
            project = self.client.find_project(project_uid)
            workspace = self.client.create_workspace(project_uid, title)
            
            return ToolResult.success(
                {"workspace_uid": workspace.uid, "title": title},
                f"Created workspace {workspace.uid}"
            )
        except Exception as e:
            return ToolResult.failure(f"Failed to create workspace: {str(e)}")
    
    def kill_job(self, job_uid: str, project_uid: str) -> ToolResult:
        """Kill a running job."""
        try:
            job = self.client.find_job(project_uid, job_uid)
            job.kill()
            
            return ToolResult.success(
                {"job_uid": job_uid, "status": "killed"},
                f"Killed job {job_uid}"
            )
        except Exception as e:
            return ToolResult.failure(f"Failed to kill job: {str(e)}")
