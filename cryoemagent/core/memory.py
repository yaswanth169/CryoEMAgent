"""Memory module for CryoEMAgent - Semantic and Episodic memory management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class JobRecord:
    """Record of a completed job."""

    job_uid: str
    job_type: str
    status: JobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_uid": self.job_uid,
            "job_type": self.job_type,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }


@dataclass
class ProcessingState:
    """Current state of the processing pipeline."""

    project_uid: str
    workspace_uid: str

    current_stage: str = "initialization"
    movies_imported: bool = False
    motion_corrected: bool = False
    ctf_estimated: bool = False
    particles_picked: bool = False
    particles_extracted: bool = False
    classes_2d_done: bool = False
    abinit_done: bool = False
    refinement_done: bool = False

    total_movies: int = 0
    total_micrographs: int = 0
    total_particles: int = 0
    current_resolution: Optional[float] = None

    active_job_uid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_uid": self.project_uid,
            "workspace_uid": self.workspace_uid,
            "current_stage": self.current_stage,
            "pipeline_status": {
                "movies_imported": self.movies_imported,
                "motion_corrected": self.motion_corrected,
                "ctf_estimated": self.ctf_estimated,
                "particles_picked": self.particles_picked,
                "particles_extracted": self.particles_extracted,
                "classes_2d_done": self.classes_2d_done,
                "abinit_done": self.abinit_done,
                "refinement_done": self.refinement_done,
            },
            "counts": {
                "movies": self.total_movies,
                "micrographs": self.total_micrographs,
                "particles": self.total_particles,
            },
            "resolution": self.current_resolution,
            "active_job": self.active_job_uid,
        }


@dataclass
class SemanticMemory:
    """Long-term domain knowledge and tool capabilities."""

    objective: str = ""

    available_tools: List[str] = field(default_factory=lambda: [
        "import_movies",
        "patch_motion_correction",
        "patch_ctf_estimation",
        "blob_picker",
        "template_picker",
        "extract_micrographs",
        "class_2d",
        "select_2d",
        "homo_abinit",
        "hetero_refine",
        "homo_refine",
        "nonuniform_refine",
        "local_refine",
    ])

    domain_knowledge: Dict[str, Any] = field(default_factory=lambda: {
        "gpcr_characteristics": {
            "typical_size_kda": 60,
            "typical_box_size": 256,
            "min_diameter_angstrom": 80,
            "max_diameter_angstrom": 150,
            "expected_symmetry": "C1",
            "target_resolution": 3.5,
        },
        "quality_thresholds": {
            "min_ctf_fit": 5.0,
            "max_ice_thickness": 1.5,
            "min_particles_per_micrograph": 10,
            "min_2d_class_quality": 0.3,
        },
        "workflow_stages": [
            "import",
            "motion_correction",
            "ctf_estimation",
            "particle_picking",
            "extraction",
            "2d_classification",
            "ab_initio",
            "refinement",
        ],
    })

    def get_next_stage(self, current: str) -> Optional[str]:
        """Get the next workflow stage."""
        stages = self.domain_knowledge["workflow_stages"]
        try:
            idx = stages.index(current)
            return stages[idx + 1] if idx + 1 < len(stages) else None
        except ValueError:
            return stages[0] if stages else None

    def to_prompt_context(self) -> str:
        """Convert semantic memory to prompt context."""
        return f"""OBJECTIVE: {self.objective}

AVAILABLE TOOLS: {', '.join(self.available_tools)}

GPCR DOMAIN KNOWLEDGE:
- Target size: ~{self.domain_knowledge['gpcr_characteristics']['typical_size_kda']} kDa
- Recommended box size: {self.domain_knowledge['gpcr_characteristics']['typical_box_size']} pixels
- Particle diameter range: {self.domain_knowledge['gpcr_characteristics']['min_diameter_angstrom']}-{self.domain_knowledge['gpcr_characteristics']['max_diameter_angstrom']} Å
- Expected symmetry: {self.domain_knowledge['gpcr_characteristics']['expected_symmetry']}
- Target resolution: {self.domain_knowledge['gpcr_characteristics']['target_resolution']} Å

QUALITY THRESHOLDS:
- Minimum CTF fit: {self.domain_knowledge['quality_thresholds']['min_ctf_fit']} Å
- Maximum ice thickness: {self.domain_knowledge['quality_thresholds']['max_ice_thickness']}
- Minimum particles per micrograph: {self.domain_knowledge['quality_thresholds']['min_particles_per_micrograph']}
"""


@dataclass
class EpisodicMemory:
    """Short-term working memory for current session."""

    state: Optional[ProcessingState] = None
    job_history: List[JobRecord] = field(default_factory=list)

    observations: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # --- new fields for quality and structured decisions ---
    quality_timeline: List[Dict[str, Any]] = field(default_factory=list)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)

    def add_observation(self, observation: str, data: Optional[Dict[str, Any]] = None):
        """Record an observation."""
        self.observations.append({
            "timestamp": datetime.now().isoformat(),
            "observation": observation,
            "data": data or {},
        })

    def add_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Record an error."""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {},
        })

    def add_job(self, record: JobRecord):
        """Add a job record to history."""
        self.job_history.append(record)

    # --- new methods ---

    def add_quality_snapshot(self, snap: Any) -> None:
        """
        Append a QualitySnapshot to the quality timeline.

        Parameters
        ----------
        snap : QualitySnapshot
            Must have a ``to_dict()`` method.
        """
        try:
            self.quality_timeline.append(snap.to_dict())
        except Exception:
            # Never crash the loop over memory operations.
            pass

    def add_decision(
        self,
        step: str,
        decision: str,
        reasoning: str,
        quality_evidence: Optional[str] = None,
    ) -> None:
        """
        Record a structured agent decision.

        Parameters
        ----------
        step : str
            Pipeline step at which the decision was made.
        decision : str
            Decision string (e.g., "CONTINUE", "ADJUST", "ESCALATE").
        reasoning : str
            LLM-provided reasoning text.
        quality_evidence : str, optional
            Summary text from the most recent quality snapshot.
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "decision": decision,
            "reasoning": reasoning,
        }
        if quality_evidence is not None:
            entry["quality_evidence"] = quality_evidence
        self.decision_log.append(entry)
        # Also keep the legacy decisions list populated for backward compat.
        self.decisions.append({
            "timestamp": entry["timestamp"],
            "decision": f"{decision} @ {step}",
            "reasoning": reasoning,
        })

    def get_quality_context(self) -> str:
        """
        Return a formatted string describing the last 3 quality snapshots.

        Returns
        -------
        str
        """
        if not self.quality_timeline:
            return "No quality assessments available yet."

        recent = self.quality_timeline[-3:]
        lines = ["=== Recent Quality Assessments ==="]
        for i, snap in enumerate(recent, start=1):
            step = snap.get("step", "unknown")
            verdict = snap.get("verdict", "?")
            ts = snap.get("timestamp", "")[:19]
            lines.append(f"\n[{i}] Step: {step} | Verdict: {verdict} | Time: {ts}")
            metrics = snap.get("metrics", {})
            if metrics:
                metric_parts = []
                for k, v in metrics.items():
                    if isinstance(v, float):
                        metric_parts.append(f"{k}={v:.3g}")
                    else:
                        metric_parts.append(f"{k}={v}")
                lines.append(f"    Metrics: {', '.join(metric_parts)}")
            issues = snap.get("issues", [])
            if issues:
                lines.append(f"    Issues: {'; '.join(issues)}")
            warnings = snap.get("warnings", [])
            if warnings:
                lines.append(f"    Warnings: {'; '.join(warnings)}")
            rec = snap.get("recommendation", "")
            if rec:
                lines.append(f"    Recommendation: {rec}")

        return "\n".join(lines)

    def get_last_completed_job(self, job_type: Optional[str] = None) -> Optional[JobRecord]:
        """Get the most recent completed job, optionally filtered by type."""
        for job in reversed(self.job_history):
            if job.status == JobStatus.COMPLETED:
                if job_type is None or job.job_type == job_type:
                    return job
        return None

    def get_recent_context(self, n_observations: int = 5, n_jobs: int = 3) -> str:
        """Get recent context for planning."""
        lines = []

        if self.state:
            lines.append(f"CURRENT STATE: {self.state.current_stage}")
            lines.append(f"  - Total micrographs: {self.state.total_micrographs}")
            lines.append(f"  - Total particles: {self.state.total_particles}")
            if self.state.current_resolution:
                lines.append(f"  - Current resolution: {self.state.current_resolution:.2f} Å")

        recent_jobs = self.job_history[-n_jobs:] if self.job_history else []
        if recent_jobs:
            lines.append("\nRECENT JOBS:")
            for job in recent_jobs:
                status = "✓" if job.status == JobStatus.COMPLETED else "✗"
                lines.append(f"  {status} {job.job_uid}: {job.job_type} ({job.status.value})")
                if job.metrics:
                    for k, v in job.metrics.items():
                        lines.append(f"      - {k}: {v}")

        recent_obs = self.observations[-n_observations:] if self.observations else []
        if recent_obs:
            lines.append("\nRECENT OBSERVATIONS:")
            for obs in recent_obs:
                lines.append(f"  - {obs['observation']}")

        recent_errors = [e for e in self.errors[-3:]] if self.errors else []
        if recent_errors:
            lines.append("\nRECENT ERRORS:")
            for err in recent_errors:
                lines.append(f"  ⚠ {err['error']}")

        return "\n".join(lines)


@dataclass
class Memory:
    """Combined memory system."""

    semantic: SemanticMemory = field(default_factory=SemanticMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)

    def initialize(self, project_uid: str, workspace_uid: str, objective: str):
        """Initialize memory for a new processing session."""
        self.semantic.objective = objective
        self.episodic.state = ProcessingState(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
        )

    def get_full_context(self) -> str:
        """Get full context for LLM planning, including quality and decision history."""
        semantic_context = self.semantic.to_prompt_context()
        episodic_context = self.episodic.get_recent_context()

        # Include recent quality timeline (last 3)
        quality_section = ""
        if self.episodic.quality_timeline:
            recent_quality = self.episodic.quality_timeline[-3:]
            quality_lines = ["\nQUALITY TIMELINE (last 3 assessments):"]
            for snap in recent_quality:
                step = snap.get("step", "?")
                verdict = snap.get("verdict", "?")
                metrics = snap.get("metrics", {})
                metric_str = ", ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()
                )
                quality_lines.append(f"  [{verdict}] {step}: {metric_str}")
                issues = snap.get("issues", [])
                if issues:
                    quality_lines.append(f"    Issues: {'; '.join(issues)}")
            quality_section = "\n".join(quality_lines)

        # Include recent decision log (last 5)
        decision_section = ""
        if self.episodic.decision_log:
            recent_decisions = self.episodic.decision_log[-5:]
            decision_lines = ["\nDECISION LOG (last 5):"]
            for entry in recent_decisions:
                ts = entry.get("timestamp", "")[:19]
                step = entry.get("step", "?")
                decision = entry.get("decision", "?")
                reasoning = entry.get("reasoning", "")[:120]
                decision_lines.append(f"  [{ts}] {step} -> {decision}: {reasoning}")
            decision_section = "\n".join(decision_lines)

        return f"""{semantic_context}

CURRENT SESSION:
{episodic_context}
{quality_section}
{decision_section}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "semantic": {
                "objective": self.semantic.objective,
                "available_tools": self.semantic.available_tools,
                "domain_knowledge": self.semantic.domain_knowledge,
            },
            "episodic": {
                "state": self.episodic.state.to_dict() if self.episodic.state else None,
                "job_history": [j.to_dict() for j in self.episodic.job_history],
                "observations": self.episodic.observations,
                "decisions": self.episodic.decisions,
                "errors": self.episodic.errors,
                "quality_timeline": self.episodic.quality_timeline,
                "decision_log": self.episodic.decision_log,
            },
        }

    def save(self, filepath: str):
        """Save memory to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
