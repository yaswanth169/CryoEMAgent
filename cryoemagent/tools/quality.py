"""Quality assessment tools for cryo-EM processing."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from cryoemagent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics from cryo-EM processing."""
    
    resolution: Optional[float] = None
    num_particles: int = 0
    num_micrographs: int = 0
    
    ctf_fit_mean: Optional[float] = None
    ctf_fit_std: Optional[float] = None
    
    ice_thickness_mean: Optional[float] = None
    
    particles_per_micrograph: float = 0.0
    
    class_distribution: Dict[int, int] = None
    
    def __post_init__(self):
        self.class_distribution = self.class_distribution or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "num_particles": self.num_particles,
            "num_micrographs": self.num_micrographs,
            "ctf_fit_mean": self.ctf_fit_mean,
            "ctf_fit_std": self.ctf_fit_std,
            "ice_thickness_mean": self.ice_thickness_mean,
            "particles_per_micrograph": self.particles_per_micrograph,
            "class_distribution": self.class_distribution,
        }


class QualityAssessment:
    """Quality assessment for cryo-EM processing results."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "min_ctf_fit": 5.0,
            "max_ice_thickness": 1.5,
            "min_particles_per_micrograph": 10,
            "min_particles_total": 10000,
            "target_resolution": 3.5,
        }
    
    def assess_ctf_estimation(self, dataset) -> ToolResult:
        """Assess CTF estimation quality."""
        try:
            ctf_fits = []
            ice_thickness = []
            
            for row in dataset.rows():
                if hasattr(row, '__getitem__'):
                    if 'ctf/ctf_fit_to_A' in row:
                        ctf_fits.append(row['ctf/ctf_fit_to_A'])
                    if 'ctf/ice_thickness_rel' in row:
                        ice_thickness.append(row['ctf/ice_thickness_rel'])
            
            if not ctf_fits:
                return ToolResult.failure("No CTF data found in dataset")
            
            import numpy as np
            ctf_mean = float(np.mean(ctf_fits))
            ctf_std = float(np.std(ctf_fits))
            ice_mean = float(np.mean(ice_thickness)) if ice_thickness else None
            
            good_ctf = sum(1 for c in ctf_fits if c <= self.thresholds["min_ctf_fit"])
            good_ratio = good_ctf / len(ctf_fits)
            
            is_acceptable = ctf_mean <= self.thresholds["min_ctf_fit"]
            
            return ToolResult.success(
                {
                    "ctf_fit_mean": ctf_mean,
                    "ctf_fit_std": ctf_std,
                    "ice_thickness_mean": ice_mean,
                    "num_micrographs": len(ctf_fits),
                    "good_micrographs": good_ctf,
                    "good_ratio": good_ratio,
                    "is_acceptable": is_acceptable,
                },
                f"CTF fit: {ctf_mean:.2f}Å (mean), {good_ratio*100:.1f}% acceptable"
            )
        except Exception as e:
            return ToolResult.failure(f"CTF assessment failed: {str(e)}")
    
    def assess_particle_picking(self, dataset, num_micrographs: int) -> ToolResult:
        """Assess particle picking quality."""
        try:
            num_particles = len(dataset)
            particles_per_mic = num_particles / max(num_micrographs, 1)
            
            is_acceptable = (
                num_particles >= self.thresholds["min_particles_total"] and
                particles_per_mic >= self.thresholds["min_particles_per_micrograph"]
            )
            
            return ToolResult.success(
                {
                    "num_particles": num_particles,
                    "num_micrographs": num_micrographs,
                    "particles_per_micrograph": particles_per_mic,
                    "is_acceptable": is_acceptable,
                },
                f"Picked {num_particles} particles ({particles_per_mic:.1f} per micrograph)"
            )
        except Exception as e:
            return ToolResult.failure(f"Particle assessment failed: {str(e)}")
    
    def assess_2d_classification(self, class_info: Dict[str, Any]) -> ToolResult:
        """Assess 2D classification quality."""
        try:
            class_counts = class_info.get("class_counts", {})
            total_particles = sum(class_counts.values())
            num_classes = len(class_counts)
            
            sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
            top_classes = sorted_classes[:10] if len(sorted_classes) >= 10 else sorted_classes
            
            top_class_ratio = sum(c[1] for c in top_classes) / max(total_particles, 1)
            
            return ToolResult.success(
                {
                    "num_classes": num_classes,
                    "total_particles": total_particles,
                    "top_class_ratio": top_class_ratio,
                    "class_distribution": dict(sorted_classes[:20]),
                },
                f"{num_classes} classes, top 10 contain {top_class_ratio*100:.1f}% of particles"
            )
        except Exception as e:
            return ToolResult.failure(f"2D classification assessment failed: {str(e)}")
    
    def assess_refinement(self, refinement_info: Dict[str, Any]) -> ToolResult:
        """Assess refinement quality."""
        try:
            resolution = refinement_info.get("resolution")
            if resolution is None:
                return ToolResult.failure("No resolution found in refinement info")
            
            is_target_reached = resolution <= self.thresholds["target_resolution"]
            
            num_particles = refinement_info.get("num_particles", 0)
            
            return ToolResult.success(
                {
                    "resolution": resolution,
                    "target_resolution": self.thresholds["target_resolution"],
                    "is_target_reached": is_target_reached,
                    "num_particles": num_particles,
                },
                f"Resolution: {resolution:.2f}Å (target: {self.thresholds['target_resolution']}Å)"
            )
        except Exception as e:
            return ToolResult.failure(f"Refinement assessment failed: {str(e)}")
    
    def get_overall_quality(self, metrics: QualityMetrics) -> ToolResult:
        """Get overall quality assessment."""
        issues = []
        warnings = []
        
        if metrics.ctf_fit_mean and metrics.ctf_fit_mean > self.thresholds["min_ctf_fit"]:
            issues.append(f"CTF fit too high: {metrics.ctf_fit_mean:.2f}Å")
        
        if metrics.ice_thickness_mean and metrics.ice_thickness_mean > self.thresholds["max_ice_thickness"]:
            warnings.append(f"High ice thickness: {metrics.ice_thickness_mean:.2f}")
        
        if metrics.num_particles < self.thresholds["min_particles_total"]:
            issues.append(f"Low particle count: {metrics.num_particles}")
        
        if metrics.particles_per_micrograph < self.thresholds["min_particles_per_micrograph"]:
            warnings.append(f"Low particles per micrograph: {metrics.particles_per_micrograph:.1f}")
        
        quality_score = 1.0
        quality_score -= len(issues) * 0.2
        quality_score -= len(warnings) * 0.1
        quality_score = max(0.0, quality_score)
        
        is_acceptable = len(issues) == 0
        
        return ToolResult.success(
            {
                "quality_score": quality_score,
                "is_acceptable": is_acceptable,
                "issues": issues,
                "warnings": warnings,
                "metrics": metrics.to_dict(),
            },
            f"Quality score: {quality_score:.2f} ({'acceptable' if is_acceptable else 'needs improvement'})"
        )
