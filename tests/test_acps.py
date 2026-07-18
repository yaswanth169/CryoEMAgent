"""Tests for the ACPS (Adaptive Closed-Loop Particle Selection) algorithm."""

import pytest
from cryoemagent.core.acps import ACPS, ACPSIteration


class TestACPSStep:
    def test_lowers_threshold_when_resolution_worse_than_target(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10)
        new = acps.step(current_resolution=5.8, current_threshold=0.35)
        assert new < 0.35, "Should lower threshold to admit more particles when resolution is poor"

    def test_raises_threshold_when_resolution_better_than_target(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10)
        new = acps.step(current_resolution=3.2, current_threshold=0.35)
        assert new > 0.35, "Should raise threshold to tighten quality when already past target"

    def test_no_change_at_exact_target(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10)
        new = acps.step(current_resolution=3.5, current_threshold=0.50)
        assert new == pytest.approx(0.50), "No adjustment needed when exactly at target"

    def test_threshold_clipped_at_minimum(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10, min_threshold=0.20)
        new = acps.step(current_resolution=100.0, current_threshold=0.21)
        assert new == pytest.approx(0.20)

    def test_threshold_clipped_at_maximum(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10, max_threshold=0.90)
        new = acps.step(current_resolution=1.0, current_threshold=0.89)
        assert new == pytest.approx(0.90)

    def test_adjustment_scales_with_alpha(self):
        acps_slow = ACPS(target_resolution=3.5, alpha=0.05)
        acps_fast = ACPS(target_resolution=3.5, alpha=0.20)
        slow_new = acps_slow.step(current_resolution=5.0, current_threshold=0.50)
        fast_new = acps_fast.step(current_resolution=5.0, current_threshold=0.50)
        assert fast_new < slow_new, "Higher alpha should produce larger threshold decrease"

    def test_history_appended_each_call(self):
        acps = ACPS(target_resolution=3.5)
        acps.step(5.8, 0.35)
        acps.step(4.9, 0.33)
        acps.step(4.1, 0.34)
        assert len(acps.history) == 3

    def test_history_record_fields(self):
        acps = ACPS(target_resolution=3.5, alpha=0.10)
        new = acps.step(current_resolution=5.8, current_threshold=0.35, particles=9400)
        record = acps.history[0]
        assert isinstance(record, ACPSIteration)
        assert record.iteration == 1
        assert record.threshold_before == pytest.approx(0.35)
        assert record.threshold_after == pytest.approx(new)
        assert record.resolution_A == pytest.approx(5.8)
        assert record.particles == 9400
        assert record.gap_A == pytest.approx(2.3)


class TestACPSConvergence:
    def test_converged_at_target(self):
        acps = ACPS(target_resolution=3.5, delta=0.15)
        assert acps.converged(3.5) is True

    def test_converged_within_delta(self):
        acps = ACPS(target_resolution=3.5, delta=0.15)
        assert acps.converged(3.6) is True
        assert acps.converged(3.4) is True

    def test_not_converged_outside_delta(self):
        acps = ACPS(target_resolution=3.5, delta=0.15)
        assert acps.converged(5.8) is False
        assert acps.converged(3.7) is False

    def test_exhausted_after_max_iterations(self):
        acps = ACPS(target_resolution=3.5, max_iterations=3)
        assert acps.exhausted() is False
        acps.step(5.8, 0.35)
        acps.step(4.9, 0.33)
        assert acps.exhausted() is False
        acps.step(4.1, 0.34)
        assert acps.exhausted() is True

    def test_should_stop_when_converged(self):
        acps = ACPS(target_resolution=3.5, delta=0.15)
        assert acps.should_stop(3.5) is True

    def test_should_stop_when_exhausted(self):
        acps = ACPS(target_resolution=3.5, max_iterations=1)
        acps.step(5.8, 0.35)
        assert acps.should_stop(5.8) is True


class TestACPSRealisticScenario:
    def test_full_convergence_scenario(self):
        """Simulate CryoWizard-comparable scenario: starts at 5.8Å, converges to ~3.5Å."""
        acps = ACPS(target_resolution=3.5, alpha=0.10, delta=0.15)
        threshold = 0.35
        resolutions = [5.8, 4.9, 4.1, 3.6]  # realistic improvement sequence

        for res in resolutions:
            threshold = acps.step(res, threshold)
            if acps.converged(res):
                break

        assert acps.converged(resolutions[-1]), "Should converge after realistic sequence"
        assert len(acps.history) <= 4
        # Final threshold should be lower than starting (more particles were needed)
        assert acps.history[-1].threshold_after <= 0.35 + 0.05

    def test_summary_nonempty_after_iterations(self):
        acps = ACPS(target_resolution=3.5)
        acps.step(5.8, 0.35)
        summary = acps.summary()
        assert "ACPS" in summary
        assert "5.80" in summary

    def test_to_dict_serializable(self):
        acps = ACPS(target_resolution=3.5)
        acps.step(5.8, 0.35)
        d = acps.to_dict()
        assert d["target_resolution_A"] == 3.5
        assert len(d["history"]) == 1
        assert "resolution_A" in d["history"][0]
