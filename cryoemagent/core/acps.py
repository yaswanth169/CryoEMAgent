"""
Adaptive Closed-Loop Particle Selection (ACPS)

Novel algorithm for CryoEMAgent. Instead of CryoWizard's fixed threshold (0.8),
ACPS dynamically adjusts the particle selection threshold based on downstream
resolution feedback in a closed feedback loop.

Control law:
    gap = current_resolution - target_resolution  (positive = worse than target)
    threshold(t+1) = threshold(t) - alpha * gap

When resolution is worse than target  → gap > 0 → threshold decreases → more particles
When resolution is better than target → gap < 0 → threshold increases → fewer, cleaner

Convergence: abs(current_resolution - target_resolution) < delta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ACPSIteration:
    """Record of one ACPS feedback iteration."""

    iteration: int
    threshold_before: float
    threshold_after: float
    resolution_A: float
    particles: int
    gap_A: float
    converged: bool


class ACPS:
    """
    Adaptive Closed-Loop Particle Selection.

    Usage:
        acps = ACPS(target_resolution=3.5)

        # After each refinement iteration:
        new_threshold = acps.step(
            current_resolution=5.8,
            current_threshold=0.35,
        )
        if acps.converged(current_resolution=5.8):
            break

    Parameters
    ----------
    target_resolution : float
        Target GS-FSC resolution in Angstroms (e.g. 3.5).
    alpha : float
        Learning rate. Controls how aggressively the threshold adjusts.
        0.05 = cautious, 0.10 = standard, 0.20 = aggressive.
    delta : float
        Convergence tolerance in Angstroms.
        Converged when |resolution - target| < delta.
    min_threshold : float
        Lower bound on threshold. Never drops below this (ensures quality floor).
    max_threshold : float
        Upper bound on threshold. Never rises above this (always keeps some particles).
    max_iterations : int
        Hard stop after this many iterations regardless of convergence.
    """

    def __init__(
        self,
        target_resolution: float = 3.5,
        alpha: float = 0.02,
        delta: float = 0.15,
        min_threshold: float = 0.20,
        max_threshold: float = 0.90,
        max_iterations: int = 10,
    ) -> None:
        self.target_resolution = target_resolution
        self.alpha = alpha
        self.delta = delta
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_iterations = max_iterations
        self.history: List[ACPSIteration] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        current_resolution: float,
        current_threshold: float,
        particles: int = 0,
    ) -> float:
        """
        Compute the next threshold given the current resolution.

        Parameters
        ----------
        current_resolution : float
            Resolution from the most recent refinement (Angstroms).
            Lower is better (3.5 Å is better than 5.0 Å).
        current_threshold : float
            The particle selection threshold used in this iteration (0–1).
        particles : int
            Number of particles in refinement (used for logging only).

        Returns
        -------
        float
            Next threshold to use. Always in [min_threshold, max_threshold].
        """
        gap = current_resolution - self.target_resolution
        adjustment = self.alpha * gap
        new_threshold = current_threshold - adjustment
        new_threshold = float(max(self.min_threshold, min(self.max_threshold, new_threshold)))

        record = ACPSIteration(
            iteration=len(self.history) + 1,
            threshold_before=current_threshold,
            threshold_after=new_threshold,
            resolution_A=current_resolution,
            particles=particles,
            gap_A=gap,
            converged=self.converged(current_resolution),
        )
        self.history.append(record)

        logger.info(
            "ACPS iter=%d  res=%.2fÅ  gap=%.2fÅ  threshold %.3f → %.3f  converged=%s",
            record.iteration,
            current_resolution,
            gap,
            current_threshold,
            new_threshold,
            record.converged,
        )
        return new_threshold

    def converged(self, current_resolution: float) -> bool:
        """Return True if resolution is within delta of target."""
        return abs(current_resolution - self.target_resolution) < self.delta

    def exhausted(self) -> bool:
        """Return True if max_iterations has been reached."""
        return len(self.history) >= self.max_iterations

    def should_stop(self, current_resolution: float) -> bool:
        """Return True if ACPS should stop (converged or exhausted)."""
        return self.converged(current_resolution) or self.exhausted()

    def summary(self) -> str:
        """One-line summary of the optimization trajectory."""
        if not self.history:
            return "ACPS: no iterations yet"
        first = self.history[0]
        last = self.history[-1]
        improvement = first.resolution_A - last.resolution_A
        return (
            f"ACPS: {len(self.history)} iters | "
            f"{first.resolution_A:.2f}A -> {last.resolution_A:.2f}A "
            f"(delta={improvement:+.2f}A) | "
            f"threshold {first.threshold_before:.3f} -> {last.threshold_after:.3f} | "
            f"converged={self.converged(last.resolution_A)}"
        )

    def to_dict(self) -> dict:
        """Serialise for logging / reasoning trace."""
        return {
            "target_resolution_A": self.target_resolution,
            "alpha": self.alpha,
            "delta": self.delta,
            "iterations": len(self.history),
            "converged": self.converged(self.history[-1].resolution_A) if self.history else False,
            "history": [
                {
                    "iteration": h.iteration,
                    "threshold_before": h.threshold_before,
                    "threshold_after": h.threshold_after,
                    "resolution_A": h.resolution_A,
                    "particles": h.particles,
                    "gap_A": h.gap_A,
                }
                for h in self.history
            ],
        }
