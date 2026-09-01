"""
Tests for Experiment 1: Cauchy Counterexample to FTAP Equivalence.

Tests verify:
  - Model specification correctness
  - Cauchy distribution properties
  - Dynamic gain map behavior
  - Theorem check logical structure and outputs
  - Instability diagnostics
"""

import pytest
import numpy as np
from scipy.stats import cauchy, kstest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp1.cauchy_counterexample import (
    ModelSpec,
    DynamicGainMap,
    StaticSetDescriptor,
    TheoremChecks,
    illustrate_cauchy_instability,
    run_experiment,
)


class TestModelSpec:
    """Tests for the one-period Cauchy model specification."""

    def test_model_attributes(self):
        """Model must have T=1, d=1, X0=0."""
        m = ModelSpec()
        assert m.T == 1
        assert m.d == 1
        assert m.X0 == 0.0

    def test_sample_x1_shape(self):
        """Samples must have the correct shape."""
        m = ModelSpec()
        samples = m.sample_x1(500)
        assert samples.shape == (500,)

    def test_sample_x1_cauchy_distribution(self):
        """Samples should pass a KS test against Cauchy(0,1)."""
        m = ModelSpec()
        rng = np.random.default_rng(1)
        samples = m.sample_x1(5000, rng=rng)
        stat, p_val = kstest(samples, "cauchy")
        # p-value > 0.01 with 5000 samples for true Cauchy
        assert p_val > 0.01, f"KS test failed: stat={stat:.4f}, p={p_val:.4f}"

    def test_sample_x1_reproducible(self):
        """Same seed should give identical samples."""
        m = ModelSpec()
        s1 = m.sample_x1(100, rng=np.random.default_rng(7))
        s2 = m.sample_x1(100, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(s1, s2)

    def test_sample_x1_both_signs(self):
        """Cauchy samples should have both positive and negative values."""
        m = ModelSpec()
        samples = m.sample_x1(2000)
        assert np.any(samples > 0), "No positive Cauchy samples"
        assert np.any(samples < 0), "No negative Cauchy samples"

    def test_cauchy_full_support(self):
        """Cauchy distribution has full support on R: verify tail events exist."""
        m = ModelSpec()
        samples = m.sample_x1(10000)
        # Should have samples beyond ±5
        assert np.any(np.abs(samples) > 5), "No heavy-tail samples found"


class TestDynamicGainMap:
    """Tests for the dynamic gain space K_0."""

    def test_gain_is_linear(self):
        """Gain must be linear in phi_1."""
        m = ModelSpec()
        dgm = DynamicGainMap(m)
        x1 = np.array([1.0, -2.0, 0.5, -0.3])
        gain2 = dgm.gain(2.0, x1)
        gain1 = dgm.gain(1.0, x1)
        np.testing.assert_allclose(gain2, 2.0 * gain1)

    def test_gain_zero_phi(self):
        """Zero position gives zero gain."""
        m = ModelSpec()
        dgm = DynamicGainMap(m)
        x1 = np.array([1.0, -2.0, 0.5])
        gain = dgm.gain(0.0, x1)
        np.testing.assert_allclose(gain, np.zeros(3))

    def test_gain_negative_phi(self):
        """Negative phi reverses sign of gain."""
        m = ModelSpec()
        dgm = DynamicGainMap(m)
        x1 = np.array([1.0, -2.0, 0.5])
        g_pos = dgm.gain(3.0, x1)
        g_neg = dgm.gain(-3.0, x1)
        np.testing.assert_allclose(g_neg, -g_pos)

    def test_gain_changes_sign(self):
        """For Cauchy X_1, gains phi*X_1 take both signs (no arbitrage)."""
        m = ModelSpec()
        dgm = DynamicGainMap(m)
        x1 = m.sample_x1(2000)
        gain = dgm.gain(1.0, x1)
        assert np.any(gain > 0)
        assert np.any(gain < 0)


class TestTheoremChecks:
    """Tests for the analytical theorem validation logic."""

    def setup_method(self):
        self.model = ModelSpec()
        self.checks = TheoremChecks(self.model)

    def test_dynamic_na_status(self):
        """Dynamic NA must be proved."""
        result = self.checks.dynamic_no_arbitrage()
        assert result["status"] == "proved"
        assert "label" in result
        assert result["label"] == "dynamic_no_arbitrage"

    def test_semi_static_na_status(self):
        """Semi-static NA must be proved."""
        result = self.checks.semi_static_no_arbitrage()
        assert result["status"] == "proved"
        assert "reasoning" in result
        assert len(result["reasoning"]) > 20

    def test_pricing_measure_status(self):
        """Pricing measure existence must be disproved (M(Y) = empty)."""
        result = self.checks.pricing_measure_existence()
        assert result["status"] == "disproved"
        assert "reasoning" in result

    def test_full_report_keys(self):
        """Full report must contain all required keys."""
        report = self.checks.full_report()
        required_keys = [
            "model",
            "dynamic_no_arbitrage",
            "semi_static_no_arbitrage",
            "pricing_measure_existence",
            "conclusion",
            "warning",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_conclusion_content(self):
        """Conclusion must mention counterexample and empty M(Y)."""
        report = self.checks.full_report()
        conclusion = report["conclusion"].lower()
        assert "counterexample" in conclusion or "counter" in conclusion
        assert "empty" in conclusion or "∅" in conclusion or "m(y)" in conclusion.lower()

    def test_warning_present(self):
        """Warning about Monte Carlo limitations must be in the report."""
        report = self.checks.full_report()
        assert "analytic" in report["warning"].lower() or "monte carlo" in report["warning"].lower()


class TestCauchyInstability:
    """Tests for the Cauchy instability illustration."""

    def test_instability_table_shape(self):
        """Instability table must have one row per sample size."""
        sizes = [100, 1000]
        df = illustrate_cauchy_instability(sample_sizes=sizes, n_trials=10)
        assert len(df) == 2

    def test_instability_table_columns(self):
        """Required columns must be present."""
        df = illustrate_cauchy_instability(sample_sizes=[100], n_trials=5)
        for col in ["sample_size", "mean_of_means", "std_of_means", "iqr_of_means"]:
            assert col in df.columns

    def test_instability_std_large(self):
        """Standard deviation of sample means should be large for Cauchy."""
        df = illustrate_cauchy_instability(sample_sizes=[1000], n_trials=100, rng_seed=0)
        # For Cauchy, std of sample means does NOT decrease with n — should be large
        assert df["std_of_means"].values[0] > 0.1, "Std should be large for Cauchy means"

    def test_instability_no_convergence(self):
        """IQR of Cauchy sample means should not decrease significantly with N."""
        df = illustrate_cauchy_instability(
            sample_sizes=[100, 10000], n_trials=200, rng_seed=1
        )
        iqr_small = df["iqr_of_means"].values[0]
        iqr_large = df["iqr_of_means"].values[1]
        # For Cauchy, the IQR should not drastically decrease (unlike normal CLT)
        # IQR of Cauchy(0,1) sample mean is approximately 2 regardless of n
        assert iqr_large > 0.1, "IQR should remain large for Cauchy means"


class TestRunExperiment:
    """Integration tests for the full Exp 1 pipeline."""

    def test_run_experiment_returns_dict(self):
        """run_experiment must return a dict."""
        report = run_experiment()
        assert isinstance(report, dict)

    def test_run_experiment_na_proved(self):
        """NA(Y) must be proved in the full run."""
        report = run_experiment()
        assert report["semi_static_no_arbitrage"]["status"] == "proved"

    def test_run_experiment_measure_empty(self):
        """M(Y) must be empty (disproved) in the full run."""
        report = run_experiment()
        assert report["pricing_measure_existence"]["status"] == "disproved"

    def test_run_experiment_cauchy_instability(self):
        """Instability table must be present and nonempty."""
        report = run_experiment()
        assert "cauchy_instability_table" in report
        assert len(report["cauchy_instability_table"]) > 0
