"""Mock CryoSPARC client for testing without a real CryoSPARC installation."""

import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from cryoemagent.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class MockJob:
    """Simulated CryoSPARC job."""
    
    uid: str
    project_uid: str
    workspace_uid: str
    job_type: str
    status: str = "building"
    title: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def queue(self, lane: Optional[str] = None, gpus: Optional[List[int]] = None):
        self.status = "queued"
        logger.info(f"[MOCK] Job {self.uid} queued on lane={lane}")
    
    def wait_for_done(self, error_on_incomplete: bool = True, timeout: Optional[int] = None) -> str:
        time.sleep(0.5)
        self.status = "completed"
        self._generate_outputs()
        logger.info(f"[MOCK] Job {self.uid} completed")
        return "completed"
    
    def refresh(self):
        pass
    
    def set_param(self, name: str, value: Any) -> bool:
        self.params[name] = value
        return True
    
    def connect(self, input_name: str, source_job: str, source_output: str) -> bool:
        self.inputs[input_name] = f"{source_job}.{source_output}"
        return True
    
    def _generate_outputs(self):
        """Generate mock outputs based on job type."""
        output_map = {
            "import_movies": {"movies": {"count": 1000, "type": "exposure"}},
            "patch_motion_correction_multi": {"micrographs": {"count": 1000, "type": "micrograph"}},
            "patch_ctf_estimation_multi": {"micrographs": {"count": 980, "type": "micrograph"}},
            "blob_picker_gpu": {"particles": {"count": 150000, "type": "particle"}},
            "extract_micrographs_multi": {"particles": {"count": 150000, "type": "particle"}},
            "class_2D_new": {"particles": {"count": 120000, "type": "particle"}, "class_averages": {"count": 50}},
            "select_2D": {"particles": {"count": 80000, "type": "particle"}},
            "homo_abinit": {"volume": {"resolution": 8.5}, "particles": {"count": 80000}},
            "hetero_refine": {"volume": {"resolution": 5.2}, "particles": {"count": 60000}},
            "homo_refine_new": {"volume": {"resolution": 4.1}, "particles": {"count": 55000}},
            "nonuniform_refine_new": {"volume": {"resolution": 3.2}, "particles": {"count": 52000}},
        }
        self.outputs = output_map.get(self.job_type, {"output": {"count": 100}})


class MockCryoSPARCClient:
    """Mock CryoSPARC client that simulates job execution."""
    
    def __init__(self, url: str = "http://mock:39000"):
        self.url = url
        self.jobs: Dict[str, MockJob] = {}
        self.job_counter = 0
        logger.info(f"[MOCK] CryoSPARC client initialized (simulation mode)")
    
    def test_connection(self) -> bool:
        logger.info(f"[MOCK] Connection test successful")
        return True
    
    def create_job(
        self,
        project_uid: str,
        workspace_uid: str,
        job_type: str,
        connections: Optional[Dict] = None,
        params: Optional[Dict] = None,
        title: str = "",
    ) -> MockJob:
        self.job_counter += 1
        job_uid = f"J{self.job_counter}"
        
        job = MockJob(
            uid=job_uid,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            job_type=job_type,
            title=title or f"Mock {job_type}",
            params=params or {},
        )
        
        self.jobs[job_uid] = job
        logger.info(f"[MOCK] Created job {job_uid}: {job_type}")
        return job
    
    def find_job(self, project_uid: str, job_uid: str) -> MockJob:
        if job_uid in self.jobs:
            return self.jobs[job_uid]
        raise ValueError(f"Job {job_uid} not found")
    
    def find_project(self, project_uid: str):
        return MockProject(project_uid, self)


@dataclass
class MockProject:
    """Mock CryoSPARC project."""
    
    uid: str
    client: MockCryoSPARCClient
    
    def find_jobs(self, **kwargs) -> List[MockJob]:
        return list(self.client.jobs.values())


class MockCryoSPARCTools:
    """Mock implementation of CryoSPARCTools for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self._client = MockCryoSPARCClient()
        self._current_job: Optional[MockJob] = None
        self.execution_log: List[Dict[str, Any]] = []
    
    @property
    def client(self) -> MockCryoSPARCClient:
        return self._client
    
    def test_connection(self) -> ToolResult:
        self._log("test_connection", {})
        return ToolResult.success(
            {"connected": True, "mode": "MOCK"},
            "[MOCK] Successfully connected to simulated CryoSPARC"
        )
    
    def create_job(
        self,
        project_uid: str,
        workspace_uid: str,
        job_type: str,
        title: str = "",
        params: Optional[Dict[str, Any]] = None,
        connections: Optional[Dict] = None,
    ) -> ToolResult:
        job = self._client.create_job(
            project_uid, workspace_uid, job_type,
            params=params, title=title
        )
        self._current_job = job
        
        self._log("create_job", {
            "job_uid": job.uid,
            "job_type": job_type,
            "params": params,
        })
        
        return ToolResult.success(
            {
                "job_uid": job.uid,
                "job_type": job_type,
                "status": job.status,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
            },
            f"[MOCK] Created job {job.uid}: {job_type}"
        )
    
    def set_param(self, job_uid: str, project_uid: str, name: str, value: Any) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        job.set_param(name, value)
        
        self._log("set_param", {"job_uid": job_uid, "name": name, "value": value})
        
        return ToolResult.success(
            {"job_uid": job_uid, "param": name, "value": value},
            f"[MOCK] Set {name}={value} on {job_uid}"
        )
    
    def connect_input(
        self,
        job_uid: str,
        project_uid: str,
        input_name: str,
        source_job_uid: str,
        source_output: str,
    ) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        job.connect(input_name, source_job_uid, source_output)
        
        self._log("connect_input", {
            "job_uid": job_uid,
            "input": input_name,
            "source": f"{source_job_uid}.{source_output}",
        })
        
        return ToolResult.success(
            {"job_uid": job_uid, "input": input_name, "source": f"{source_job_uid}.{source_output}"},
            f"[MOCK] Connected {input_name} to {source_job_uid}.{source_output}"
        )
    
    def queue_job(
        self,
        job_uid: str,
        project_uid: str,
        lane: Optional[str] = None,
        gpus: Optional[List[int]] = None,
    ) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        job.queue(lane=lane, gpus=gpus)
        
        self._log("queue_job", {"job_uid": job_uid, "lane": lane})
        
        return ToolResult.success(
            {"job_uid": job_uid, "status": job.status, "lane": lane},
            f"[MOCK] Queued job {job_uid}"
        )
    
    def wait_for_job(
        self,
        job_uid: str,
        project_uid: str,
        timeout: Optional[int] = None,
        error_on_incomplete: bool = True,
    ) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        status = job.wait_for_done(error_on_incomplete=error_on_incomplete, timeout=timeout)
        
        self._log("wait_for_job", {
            "job_uid": job_uid,
            "status": status,
            "outputs": job.outputs,
        })
        
        return ToolResult.success(
            {
                "job_uid": job_uid,
                "status": status,
                "outputs": job.outputs,
                "job_type": job.job_type,
            },
            f"[MOCK] Job {job_uid} completed with status: {status}"
        )
    
    def load_output(
        self,
        job_uid: str,
        project_uid: str,
        output_name: str,
    ) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        output = job.outputs.get(output_name, {"count": 0})
        
        self._log("load_output", {
            "job_uid": job_uid,
            "output_name": output_name,
            "output": output,
        })
        
        return ToolResult.success(
            {
                "job_uid": job_uid,
                "output_name": output_name,
                "num_rows": output.get("count", 0),
                "fields": ["uid", "location", "ctf/df1_A"],
            },
            f"[MOCK] Loaded {output.get('count', 0)} items from {output_name}"
        )
    
    def get_job_status(self, job_uid: str, project_uid: str) -> ToolResult:
        job = self._client.find_job(project_uid, job_uid)
        return ToolResult.success(
            {"job_uid": job_uid, "status": job.status, "type": job.job_type},
            f"[MOCK] Job {job_uid} status: {job.status}"
        )
    
    def find_jobs(
        self,
        project_uid: str,
        workspace_uid: Optional[str] = None,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> ToolResult:
        jobs = list(self._client.jobs.values())
        job_list = [
            {"uid": j.uid, "type": j.job_type, "status": j.status, "title": j.title}
            for j in jobs
        ]
        return ToolResult.success(
            {"jobs": job_list, "count": len(job_list)},
            f"[MOCK] Found {len(job_list)} jobs"
        )
    
    def _log(self, action: str, data: Dict[str, Any]):
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
        })
    
    def get_execution_summary(self) -> str:
        """Get a summary of all executed actions."""
        lines = ["=" * 60, "MOCK EXECUTION SUMMARY", "=" * 60]
        for i, entry in enumerate(self.execution_log, 1):
            lines.append(f"\n{i}. {entry['action']}")
            for k, v in entry['data'].items():
                lines.append(f"   {k}: {v}")
        return "\n".join(lines)
