"""Quality critic chain for autonomous cryo-EM processing assessment."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level quality thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, Any] = {
    "ctf": {
        "max_mean_ctf_A": 5.0,          # Mean CTF fit must be <= 5 Å to pass
        "min_fraction_ok": 0.70,         # At least 70% micrographs must pass
        "max_ice_thickness_rel": 1.2,    # Relative ice thickness upper bound
    },
    "picking": {
        "min_particles_per_mic": 50,     # Particles per micrograph (FAIL below)
        "min_total_particles": 5000,     # Total particles (WARN below)
    },
    "class2d": {
        "max_empty_frac": 0.50,          # Fraction of empty classes (WARN above)
        "max_gini": 0.85,                # Gini coefficient (WARN above)
    },
    "refinement": {
        "target_resolution_A": 3.5,      # Target resolution in Angstroms
        "warn_resolution_A": 5.0,        # Warn if above this
    },
}


# ---------------------------------------------------------------------------
# QualitySnapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class QualitySnapshot:
    """Immutable snapshot of quality assessment for a single pipeline step."""

    step: str
    verdict: str                         # "PASS", "WARN", or "FAIL"
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "step": self.step,
            "verdict": self.verdict,
            "metrics": self.metrics,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        parts = [f"[{self.verdict}] {self.step}"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.3g}")
            else:
                parts.append(f"{k}={v}")
        if self.issues:
            parts.append(f"ISSUES: {'; '.join(self.issues)}")
        if self.warnings:
            parts.append(f"WARNS: {'; '.join(self.warnings)}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# CTF Critic
# ---------------------------------------------------------------------------

class CTFCritic:
    """Assess CTF estimation quality from a cryosparc_tools job output."""

    @staticmethod
    def assess(cs_client: Any, project_uid: str, ctf_job_uid: str) -> QualitySnapshot:
        """
        Load exposures output from the CTF job and compute quality metrics.

        All exceptions are caught and converted to WARN snapshots.
        """
        step = "patch_ctf"
        try:
            job = cs_client.find_job(project_uid, ctf_job_uid)
            dataset = job.load_output("exposures")

            ctf_fits: List[float] = []
            ice_values: List[float] = []

            for row in dataset.rows():
                try:
                    ctf_val = row["ctf/ctf_fit_to_A"]
                    ctf_fits.append(float(ctf_val))
                except (KeyError, TypeError, ValueError):
                    pass
                try:
                    ice_val = row["ctf/ice_thickness_rel"]
                    ice_values.append(float(ice_val))
                except (KeyError, TypeError, ValueError):
                    pass

            if not ctf_fits:
                return QualitySnapshot(
                    step=step,
                    verdict="WARN",
                    warnings=["No CTF fit values found in exposures output"],
                    recommendation="Check CTF estimation job completed correctly.",
                )

            mean_ctf = sum(ctf_fits) / len(ctf_fits)
            variance = sum((x - mean_ctf) ** 2 for x in ctf_fits) / max(len(ctf_fits) - 1, 1)
            std_ctf = variance ** 0.5
            threshold_ctf = THRESHOLDS["ctf"]["max_mean_ctf_A"]
            fraction_ok = sum(1 for c in ctf_fits if c <= threshold_ctf) / len(ctf_fits)
            mean_ice = sum(ice_values) / len(ice_values) if ice_values else None

            metrics: Dict[str, Any] = {
                "mean_ctf_A": round(mean_ctf, 3),
                "std_ctf_A": round(std_ctf, 3),
                "fraction_ok": round(fraction_ok, 3),
                "n_micrographs": len(ctf_fits),
            }
            if mean_ice is not None:
                metrics["mean_ice_thickness_rel"] = round(mean_ice, 3)

            issues: List[str] = []
            warnings: List[str] = []

            if mean_ctf > THRESHOLDS["ctf"]["max_mean_ctf_A"]:
                issues.append(
                    f"Mean CTF fit {mean_ctf:.2f} Å exceeds threshold "
                    f"{THRESHOLDS['ctf']['max_mean_ctf_A']} Å"
                )

            if fraction_ok < THRESHOLDS["ctf"]["min_fraction_ok"]:
                warnings.append(
                    f"Only {fraction_ok * 100:.1f}% micrographs pass CTF threshold "
                    f"(minimum {THRESHOLDS['ctf']['min_fraction_ok'] * 100:.0f}%)"
                )

            if mean_ice is not None and mean_ice > THRESHOLDS["ctf"]["max_ice_thickness_rel"]:
                warnings.append(
                    f"Mean relative ice thickness {mean_ice:.2f} exceeds "
                    f"threshold {THRESHOLDS['ctf']['max_ice_thickness_rel']}"
                )

            if issues:
                verdict = "FAIL"
                recommendation = (
                    "CTF quality is poor. Consider discarding thin/thick-ice micrographs "
                    "during curation or adjusting CTF estimation parameters."
                )
            elif warnings:
                verdict = "WARN"
                recommendation = (
                    "CTF quality is marginal. Review curate-exposures step carefully."
                )
            else:
                verdict = "PASS"
                recommendation = "CTF estimation quality is acceptable. Proceed."

            return QualitySnapshot(
                step=step,
                verdict=verdict,
                metrics=metrics,
                issues=issues,
                warnings=warnings,
                recommendation=recommendation,
            )

        except Exception as exc:
            logger.warning("CTFCritic.assess failed: %s", exc, exc_info=True)
            return QualitySnapshot(
                step=step,
                verdict="WARN",
                warnings=[f"CTF quality assessment failed: {exc}"],
                recommendation="Unable to assess CTF quality automatically. Review manually.",
            )


# ---------------------------------------------------------------------------
# Picking Critic
# ---------------------------------------------------------------------------

class PickingCritic:
    """Assess particle picking quality from inspect_picks job output."""

    @staticmethod
    def assess(cs_client: Any, project_uid: str, inspect_job_uid: str) -> QualitySnapshot:
        """
        Load particle and micrograph outputs and compute picking metrics.

        All exceptions are caught and converted to WARN snapshots.
        """
        step = "inspect_picks"
        try:
            job = cs_client.find_job(project_uid, inspect_job_uid)

            # Try loading particles output
            particles_dataset = None
            for output_name in ("particles", "particles_selected"):
                try:
                    particles_dataset = job.load_output(output_name)
                    break
                except Exception:
                    continue

            if particles_dataset is None:
                return QualitySnapshot(
                    step=step,
                    verdict="WARN",
                    warnings=["Could not load particles output from inspect job"],
                    recommendation="Verify inspect picks job completed and output is available.",
                )

            # Count particles
            try:
                total_particles = len(particles_dataset)
            except Exception:
                # Fallback: iterate rows
                total_particles = sum(1 for _ in particles_dataset.rows())

            # Try loading micrograph output for count
            n_micrographs = 1  # avoid division by zero
            try:
                mic_dataset = job.load_output("micrographs")
                try:
                    n_micrographs = max(len(mic_dataset), 1)
                except Exception:
                    n_micrographs = max(sum(1 for _ in mic_dataset.rows()), 1)
            except Exception:
                pass

            particles_per_mic = total_particles / n_micrographs

            metrics: Dict[str, Any] = {
                "total_particles": total_particles,
                "n_micrographs": n_micrographs,
                "particles_per_mic": round(particles_per_mic, 2),
            }

            issues: List[str] = []
            warnings: List[str] = []

            min_per_mic = THRESHOLDS["picking"]["min_particles_per_mic"]
            min_total = THRESHOLDS["picking"]["min_total_particles"]

            if particles_per_mic < min_per_mic:
                issues.append(
                    f"Only {particles_per_mic:.1f} particles/micrograph "
                    f"(minimum required: {min_per_mic})"
                )

            if total_particles < min_total:
                warnings.append(
                    f"Low total particle count: {total_particles} "
                    f"(recommended minimum: {min_total})"
                )

            if issues:
                verdict = "FAIL"
                recommendation = (
                    "Particle yield is critically low. "
                    "Adjust picking thresholds or particle diameter parameters."
                )
            elif warnings:
                verdict = "WARN"
                recommendation = (
                    "Particle count is lower than ideal. "
                    "Consider relaxing picking parameters to increase yield."
                )
            else:
                verdict = "PASS"
                recommendation = "Particle picking yield is acceptable. Proceed."

            return QualitySnapshot(
                step=step,
                verdict=verdict,
                metrics=metrics,
                issues=issues,
                warnings=warnings,
                recommendation=recommendation,
            )

        except Exception as exc:
            logger.warning("PickingCritic.assess failed: %s", exc, exc_info=True)
            return QualitySnapshot(
                step=step,
                verdict="WARN",
                warnings=[f"Picking quality assessment failed: {exc}"],
                recommendation="Unable to assess picking quality automatically. Review manually.",
            )


# ---------------------------------------------------------------------------
# 2D Classification Critic
# ---------------------------------------------------------------------------

class Class2DCritic:
    """Assess 2D classification quality from a class2d job output."""

    @staticmethod
    def _gini(values: List[float]) -> float:
        """Compute Gini coefficient for a list of non-negative values."""
        if not values or sum(values) == 0:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = 0.0
        gini_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumsum += v
            gini_sum += (2 * (i + 1) - n - 1) * v
        total = sum(sorted_vals)
        return gini_sum / (n * total) if total else 0.0

    @staticmethod
    def assess(cs_client: Any, project_uid: str, class2d_job_uid: str) -> QualitySnapshot:
        """
        Load class averages and compute occupancy-based quality metrics.

        All exceptions are caught and converted to WARN snapshots.
        """
        step = "class2d"
        try:
            job = cs_client.find_job(project_uid, class2d_job_uid)

            # Try loading class averages output
            class_dataset = None
            for output_name in ("class_averages", "templates"):
                try:
                    class_dataset = job.load_output(output_name)
                    break
                except Exception:
                    continue

            if class_dataset is None:
                return QualitySnapshot(
                    step=step,
                    verdict="WARN",
                    warnings=["Could not load class averages output from class2d job"],
                    recommendation="Verify class2d job completed and output is available.",
                )

            occupancies: List[float] = []
            for row in class_dataset.rows():
                val = None
                for field_name in ("class2D/num_particles_total", "num_particles"):
                    try:
                        val = float(row[field_name])
                        break
                    except (KeyError, TypeError, ValueError):
                        continue
                if val is not None:
                    occupancies.append(val)

            if not occupancies:
                return QualitySnapshot(
                    step=step,
                    verdict="WARN",
                    warnings=["No occupancy data found in class averages"],
                    recommendation="Verify class2d output fields are correctly named.",
                )

            total = sum(occupancies)
            n_classes = len(occupancies)
            sorted_occ = sorted(occupancies, reverse=True)

            # Top 10 fraction
            top_10 = sorted_occ[:10]
            top10_frac = sum(top_10) / total if total > 0 else 0.0

            # Empty class fraction (zero or near-zero occupancy)
            empty_count = sum(1 for o in occupancies if o < 1.0)
            empty_frac = empty_count / n_classes if n_classes > 0 else 0.0

            gini = Class2DCritic._gini(occupancies)

            metrics: Dict[str, Any] = {
                "n_classes": n_classes,
                "total_particles": int(total),
                "top10_frac": round(top10_frac, 3),
                "empty_frac": round(empty_frac, 3),
                "gini_coefficient": round(gini, 3),
            }

            issues: List[str] = []
            warnings: List[str] = []

            max_empty = THRESHOLDS["class2d"]["max_empty_frac"]
            max_gini = THRESHOLDS["class2d"]["max_gini"]

            if empty_frac > max_empty:
                warnings.append(
                    f"High empty class fraction: {empty_frac * 100:.1f}% "
                    f"(threshold: {max_empty * 100:.0f}%)"
                )

            if gini > max_gini:
                warnings.append(
                    f"High Gini coefficient {gini:.3f} suggests uneven class distribution "
                    f"(threshold: {max_gini})"
                )

            if issues:
                verdict = "FAIL"
                recommendation = "2D classification quality is poor. Re-run with adjusted parameters."
            elif warnings:
                verdict = "WARN"
                recommendation = (
                    "2D classification quality is marginal. "
                    "Be selective when choosing classes to proceed with."
                )
            else:
                verdict = "PASS"
                recommendation = "2D classification quality is acceptable. Proceed with selection."

            return QualitySnapshot(
                step=step,
                verdict=verdict,
                metrics=metrics,
                issues=issues,
                warnings=warnings,
                recommendation=recommendation,
            )

        except Exception as exc:
            logger.warning("Class2DCritic.assess failed: %s", exc, exc_info=True)
            return QualitySnapshot(
                step=step,
                verdict="WARN",
                warnings=[f"2D classification quality assessment failed: {exc}"],
                recommendation="Unable to assess 2D classification quality automatically. Review manually.",
            )


# ---------------------------------------------------------------------------
# Refinement Critic
# ---------------------------------------------------------------------------

class RefinementCritic:
    """Assess refinement quality from a homo_refine or nonuniform_refine job."""

    @staticmethod
    def assess(
        cs_client: Any,
        project_uid: str,
        refine_job_uid: str,
        step_name: str = "refinement",
    ) -> QualitySnapshot:
        """
        Read resolution from job metadata and particle count from output.

        All exceptions are caught and converted to WARN snapshots.
        """
        try:
            job = cs_client.find_job(project_uid, refine_job_uid)

            resolution: Optional[float] = None

            # Try reading resolution from job.doc dict
            doc = None
            for attr in ("doc", "_doc", "document"):
                try:
                    doc = getattr(job, attr, None)
                    if doc is not None:
                        break
                except Exception:
                    continue

            if doc is not None:
                for key in (
                    "summary_stats",
                    "stats",
                    "result_summary",
                    "results",
                ):
                    section = None
                    try:
                        if isinstance(doc, dict):
                            section = doc.get(key, {})
                        else:
                            section = getattr(doc, key, None)
                    except Exception:
                        continue
                    if section:
                        for res_key in ("fsc_resolution_halfmap", "fsc_resolution", "resolution"):
                            try:
                                if isinstance(section, dict):
                                    val = section.get(res_key)
                                else:
                                    val = getattr(section, res_key, None)
                                if val is not None:
                                    resolution = float(val)
                                    break
                            except Exception:
                                continue
                    if resolution is not None:
                        break

            # Fallback: try job attributes directly
            if resolution is None:
                for attr_name in (
                    "fsc_resolution",
                    "resolution",
                    "fsc_resolution_halfmap",
                ):
                    try:
                        val = getattr(job, attr_name, None)
                        if val is not None:
                            resolution = float(val)
                            break
                    except Exception:
                        continue

            # Try to load particles for count
            particle_count: Optional[int] = None
            try:
                particles_dataset = job.load_output("particles")
                try:
                    particle_count = len(particles_dataset)
                except Exception:
                    particle_count = sum(1 for _ in particles_dataset.rows())
            except Exception:
                pass

            metrics: Dict[str, Any] = {}
            if resolution is not None:
                metrics["resolution_A"] = round(resolution, 2)
            if particle_count is not None:
                metrics["particle_count"] = particle_count

            issues: List[str] = []
            warnings: List[str] = []

            target = THRESHOLDS["refinement"]["target_resolution_A"]
            warn_thresh = THRESHOLDS["refinement"]["warn_resolution_A"]

            if resolution is None:
                warnings.append("Could not determine resolution from job metadata")
                verdict = "WARN"
                recommendation = "Check refinement job output and review resolution manually."
            elif resolution > warn_thresh:
                issues.append(
                    f"Resolution {resolution:.2f} Å is worse than warning threshold {warn_thresh} Å"
                )
                verdict = "FAIL"
                recommendation = (
                    f"Resolution {resolution:.2f} Å is far from target {target} Å. "
                    "Consider increasing particle count or adjusting refinement parameters."
                )
            elif resolution > target:
                warnings.append(
                    f"Resolution {resolution:.2f} Å has not reached target {target} Å"
                )
                verdict = "WARN"
                recommendation = (
                    f"Resolution {resolution:.2f} Å is close to target {target} Å. "
                    "Consider non-uniform refinement or local refinement to improve."
                )
            else:
                verdict = "PASS"
                recommendation = f"Target resolution {target} Å achieved at {resolution:.2f} Å."

            return QualitySnapshot(
                step=step_name,
                verdict=verdict,
                metrics=metrics,
                issues=issues,
                warnings=warnings,
                recommendation=recommendation,
            )

        except Exception as exc:
            logger.warning("RefinementCritic.assess failed: %s", exc, exc_info=True)
            return QualitySnapshot(
                step=step_name,
                verdict="WARN",
                warnings=[f"Refinement quality assessment failed: {exc}"],
                recommendation="Unable to assess refinement quality automatically. Review manually.",
            )


# ---------------------------------------------------------------------------
# QualityCriticChain - dispatcher
# ---------------------------------------------------------------------------

class QualityCriticChain:
    """
    Routes pipeline steps to the appropriate quality critic.

    Steps not covered by any critic return None (no assessment available).
    """

    # Map of step name patterns -> (critic_class, job_key_in_state)
    # job_key_in_state is the key in state.jobs dict that holds the job UID.
    _CTF_STEPS = {"patch_ctf"}
    _PICKING_STEPS = {"inspect_blob", "inspect_template"}
    _CLASS2D_STEPS = {"class2d_blob", "class2d_template"}
    _REFINEMENT_STEPS = {
        "homo_blob",
        "homo_template",
        "nonuniform_template",
        "nonuniform_blob",
    }

    def assess_step(
        self,
        step_name: str,
        cs_client: Any,
        project_uid: str,
        jobs_dict: Dict[str, str],
    ) -> Optional[QualitySnapshot]:
        """
        Route a pipeline step to the correct critic.

        Parameters
        ----------
        step_name : str
            The current pipeline step name (e.g., "patch_ctf", "inspect_blob").
        cs_client : CryoSPARC
            The live cryosparc-tools client.
        project_uid : str
            The CryoSPARC project UID.
        jobs_dict : dict
            Mapping of step name -> job UID from the RunState.

        Returns
        -------
        QualitySnapshot or None if no critic covers this step.
        """
        try:
            if step_name in self._CTF_STEPS:
                job_uid = jobs_dict.get(step_name)
                if not job_uid:
                    return None
                return CTFCritic.assess(cs_client, project_uid, job_uid)

            elif step_name in self._PICKING_STEPS:
                job_uid = jobs_dict.get(step_name)
                if not job_uid:
                    return None
                return PickingCritic.assess(cs_client, project_uid, job_uid)

            elif step_name in self._CLASS2D_STEPS:
                job_uid = jobs_dict.get(step_name)
                if not job_uid:
                    return None
                return Class2DCritic.assess(cs_client, project_uid, job_uid)

            elif step_name in self._REFINEMENT_STEPS:
                job_uid = jobs_dict.get(step_name)
                if not job_uid:
                    return None
                return RefinementCritic.assess(
                    cs_client, project_uid, job_uid, step_name=step_name
                )

            else:
                # No critic for this step
                return None

        except Exception as exc:
            logger.warning(
                "QualityCriticChain.assess_step raised unexpectedly for step %s: %s",
                step_name,
                exc,
                exc_info=True,
            )
            return QualitySnapshot(
                step=step_name,
                verdict="WARN",
                warnings=[f"Quality chain dispatcher error: {exc}"],
                recommendation="Unexpected error in quality assessment. Continuing.",
            )
