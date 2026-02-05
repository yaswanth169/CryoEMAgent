"""GPCR-specific workflow playbooks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class WorkflowStage(str, Enum):
    """GPCR workflow stages."""
    
    IMPORT = "import"
    MOTION_CORRECTION = "motion_correction"
    CTF_ESTIMATION = "ctf_estimation"
    DENOISING = "denoising"
    JUNK_DETECTION = "junk_detection"
    PARTICLE_PICKING = "particle_picking"
    EXTRACTION = "extraction"
    CLASSIFICATION_2D = "classification_2d"
    SELECTION_2D = "selection_2d"
    DECOY_CLASSIFICATION = "decoy_classification"
    AB_INITIO = "ab_initio"
    HETERO_REFINEMENT = "hetero_refinement"
    HOMO_REFINEMENT = "homo_refinement"
    NONUNIFORM_REFINEMENT = "nonuniform_refinement"
    LOCAL_REFINEMENT = "local_refinement"


@dataclass
class GPCRWorkflowStep:
    """Single step in GPCR workflow."""
    
    name: str
    stage: WorkflowStage
    job_type: str
    
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    required: bool = True
    condition: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage.value,
            "job_type": self.job_type,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "required": self.required,
            "condition": self.condition,
        }


@dataclass
class GPCRParameters:
    """GPCR-specific processing parameters."""
    
    min_particle_diameter: float = 80.0
    max_particle_diameter: float = 150.0
    box_size: int = 256
    fourier_crop_size: int = 128
    num_2d_classes: int = 50
    num_abinit_classes: int = 3
    num_hetero_classes: int = 4
    symmetry: str = "C1"
    resolution_target: float = 3.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_particle_diameter": self.min_particle_diameter,
            "max_particle_diameter": self.max_particle_diameter,
            "box_size": self.box_size,
            "fourier_crop_size": self.fourier_crop_size,
            "num_2d_classes": self.num_2d_classes,
            "num_abinit_classes": self.num_abinit_classes,
            "num_hetero_classes": self.num_hetero_classes,
            "symmetry": self.symmetry,
            "resolution_target": self.resolution_target,
        }


class GPCRPlaybook:
    """GPCR structure determination playbook."""
    
    def __init__(self, params: Optional[GPCRParameters] = None):
        self.params = params or GPCRParameters()
        self.steps = self._build_standard_workflow()
    
    def _build_standard_workflow(self) -> List[GPCRWorkflowStep]:
        """Build the standard GPCR workflow."""
        return [
            GPCRWorkflowStep(
                name="Import Movies",
                stage=WorkflowStage.IMPORT,
                job_type="import_movies",
                outputs=["movies"],
            ),
            GPCRWorkflowStep(
                name="Patch Motion Correction",
                stage=WorkflowStage.MOTION_CORRECTION,
                job_type="patch_motion_correction",
                inputs=["Import Movies.movies"],
                outputs=["micrographs"],
            ),
            GPCRWorkflowStep(
                name="Patch CTF Estimation",
                stage=WorkflowStage.CTF_ESTIMATION,
                job_type="patch_ctf_estimation",
                inputs=["Patch Motion Correction.micrographs"],
                outputs=["micrographs"],
            ),
            GPCRWorkflowStep(
                name="Blob Picker",
                stage=WorkflowStage.PARTICLE_PICKING,
                job_type="blob_picker",
                inputs=["Patch CTF Estimation.micrographs"],
                outputs=["particles"],
                params={
                    "min_particle_diameter": self.params.min_particle_diameter,
                    "max_particle_diameter": self.params.max_particle_diameter,
                },
            ),
            GPCRWorkflowStep(
                name="Extract Micrographs",
                stage=WorkflowStage.EXTRACTION,
                job_type="extract_micrographs",
                inputs=["Blob Picker.particles", "Patch CTF Estimation.micrographs"],
                outputs=["particles"],
                params={
                    "box_size_pix": self.params.box_size,
                },
            ),
            GPCRWorkflowStep(
                name="2D Classification",
                stage=WorkflowStage.CLASSIFICATION_2D,
                job_type="class_2d",
                inputs=["Extract Micrographs.particles"],
                outputs=["particles", "class_averages"],
                params={
                    "num_classes": self.params.num_2d_classes,
                },
            ),
            GPCRWorkflowStep(
                name="Select 2D",
                stage=WorkflowStage.SELECTION_2D,
                job_type="select_2d",
                inputs=["2D Classification.particles", "2D Classification.class_averages"],
                outputs=["particles"],
            ),
            GPCRWorkflowStep(
                name="Ab-initio Reconstruction",
                stage=WorkflowStage.AB_INITIO,
                job_type="homo_abinit",
                inputs=["Select 2D.particles"],
                outputs=["volume", "particles"],
                params={
                    "abinit_K": self.params.num_abinit_classes,
                    "abinit_symmetry": self.params.symmetry,
                },
            ),
            GPCRWorkflowStep(
                name="Heterogeneous Refinement",
                stage=WorkflowStage.HETERO_REFINEMENT,
                job_type="hetero_refine",
                inputs=["Ab-initio Reconstruction.particles", "Ab-initio Reconstruction.volume"],
                outputs=["volume", "particles"],
                params={
                    "hetero_num_classes": self.params.num_hetero_classes,
                },
            ),
            GPCRWorkflowStep(
                name="Homogeneous Refinement",
                stage=WorkflowStage.HOMO_REFINEMENT,
                job_type="homo_refine",
                inputs=["Heterogeneous Refinement.particles", "Heterogeneous Refinement.volume"],
                outputs=["volume", "particles"],
                params={
                    "refine_symmetry": self.params.symmetry,
                },
            ),
            GPCRWorkflowStep(
                name="Non-uniform Refinement",
                stage=WorkflowStage.NONUNIFORM_REFINEMENT,
                job_type="nonuniform_refine",
                inputs=["Homogeneous Refinement.particles", "Homogeneous Refinement.volume"],
                outputs=["volume", "particles"],
                params={
                    "refine_symmetry": self.params.symmetry,
                },
            ),
        ]
    
    def get_step(self, name: str) -> Optional[GPCRWorkflowStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_step_by_stage(self, stage: WorkflowStage) -> List[GPCRWorkflowStep]:
        """Get all steps for a stage."""
        return [s for s in self.steps if s.stage == stage]
    
    def get_next_step(self, current_step: str) -> Optional[GPCRWorkflowStep]:
        """Get the next step after the current one."""
        for i, step in enumerate(self.steps):
            if step.name == current_step and i + 1 < len(self.steps):
                return self.steps[i + 1]
        return None
    
    def get_connections(self, step: GPCRWorkflowStep) -> List[Tuple[str, str, str]]:
        """Get job connections for a step as (input_name, source_job, source_output) tuples."""
        connections = []
        for input_spec in step.inputs:
            parts = input_spec.split(".")
            if len(parts) == 2:
                source_job_name, source_output = parts
                connections.append((step.job_type + "_input", source_job_name, source_output))
        return connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playbook to dictionary."""
        return {
            "name": "GPCR Standard Workflow",
            "version": "1.0",
            "params": self.params.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }


def get_abinitio_playbook(params: Optional[GPCRParameters] = None) -> GPCRPlaybook:
    """Get playbook optimized for ab-initio reconstruction."""
    params = params or GPCRParameters()
    params.num_abinit_classes = 4
    return GPCRPlaybook(params)


def get_high_resolution_playbook(params: Optional[GPCRParameters] = None) -> GPCRPlaybook:
    """Get playbook optimized for high-resolution refinement."""
    params = params or GPCRParameters()
    params.box_size = 320
    params.fourier_crop_size = 160
    params.resolution_target = 2.5
    
    playbook = GPCRPlaybook(params)
    
    playbook.steps.append(GPCRWorkflowStep(
        name="Local Refinement",
        stage=WorkflowStage.LOCAL_REFINEMENT,
        job_type="local_refine",
        inputs=["Non-uniform Refinement.particles", "Non-uniform Refinement.volume"],
        outputs=["volume", "particles"],
        required=False,
    ))
    
    return playbook
