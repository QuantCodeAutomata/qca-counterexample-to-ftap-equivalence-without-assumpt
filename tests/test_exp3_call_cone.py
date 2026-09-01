"""
Tests for Experiment 3: Call-Option Small Cone and FTAP Validation.

Tests verify:
  - Lognormal model specification
  - Black-Scholes call pricing formula
  - Centered call payoff properties (E_P[h_K]=0)
  - Small-cone L^0 boundedness
  - Assumption 3.3 and nonredundancy logic
  - Pricing measure verification (P in M)
  - Theorem 3.5 implication checks
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp3.call_cone import (
    CallConeModelSpec,
    CallGeneratorFamily,
    DynamicGainSpace,
    Assumption33Checker,
    NonredundancyChecker,
    PricingMeasureFeasibility,
    ClosednessReport,
    run_experiment,
)


class TestCallConeModelSpec:
    """Tests for the lognormal call-cone model."""

    def test_default_parameters(self):
        """Default model must have c=1, sigma=0.3, K_flat=0.8, K_sharp=1.2."""
        m = CallConeModelSpec()
        assert m.c == 1.0
        assert m.sigma == 0.3
        assert m.K_flat == 0.8
        assert m.K_sharp == 1.2

    def test_invalid_parameters_raise(self):
        """Invalid parameters must raise AssertionError."""
        with pytest.raises(AssertionError):
            CallConeModelSpec(c=-1.0)
        with pytest.raises(AssertionError):
            CallConeModelSpec(sigma=-0.1)
        with pytest.raises(AssertionError):
            CallConeModelSpec(K_flat=1.5, K_sharp=1.0)

    def test_sample_x1_positive(self):
        """Lognormal samples must all be positive."""
        m = CallConeModelSpec()
        samples = m.sample_x1(5000)
        assert np.all(samples > 0), "Lognormal samples must be positive"

    def test_sample_x1_mean_close_to_c(self):
        """E_P[X_1] should be close to c by construction."""
        m = CallConeModelSpec(c=1.0, sigma=0.2)
        rng = np.random.default_rng(42)
        samples = m.sample_x1(200_000, rng=rng)
        empirical_mean = np.mean(samples)
        assert abs(empirical_mean - m.c) < 0.01, (
            f"Empirical mean {empirical_mean:.4f} far from c={m.c}"
        )

    def test_call_price_nonnegative(self):
        """Call prices E_P[(X_1-K)^+] must be nonneg."""
        m = CallConeModelSpec()
        for K in [0.5, 0.8, 1.0, 1.2, 1.5]:
            price = m.call_price_analytical(K)
            assert price >= 0.0, f"Call price negative at K={K}"

    def test_call_price_decreasing_in_K(self):
        """Call price must be decreasing in K."""
        m = CallConeModelSpec()
        strikes = [0.7, 0.9, 1.0, 1.1, 1.3]
        prices = [m.call_price_analytical(K) for K in strikes]
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1], "Call price must decrease with K"

    def test_call_price_atm(self):
        """ATM call price should be positive."""
        m = CallConeModelSpec(c=1.0, sigma=0.3)
        price_atm = m.call_price_analytical(K=1.0)
        assert price_atm > 0.01, "ATM call price should be substantial"

    def test_call_price_deep_otm(self):
        """Deep OTM call price should be near zero."""
        m = CallConeModelSpec(c=1.0, sigma=0.3)
        price_deep_otm = m.call_price_analytical(K=5.0)
        assert price_deep_otm < 0.01

    def test_centered_call_mean_near_zero(self):
        """Centered call payoff must have near-zero empirical mean."""
        m = CallConeModelSpec()
        rng = np.random.default_rng(10)
        x1 = m.sample_x1(200_000, rng=rng)
        for K in [0.8, 1.0, 1.2]:
            h_K = m.centered_call(K, x1)
            mean_h = np.mean(h_K)
            assert abs(mean_h) < 0.01, (
                f"|E[h_K]| = {abs(mean_h):.4f} at K={K} — should be ~0"
            )

    def test_call_price_grid_shape(self):
        """Call price grid DataFrame must have correct columns and rows."""
        m = CallConeModelSpec()
        df = m.call_price_grid(n_strikes=50)
        assert "K" in df.columns
        assert "call_price" in df.columns
        assert len(df) == 50


class TestCallGeneratorFamily:
    """Tests for the centered call generator family H_1."""

    def setup_method(self):
        self.model = CallConeModelSpec()
        self.fam = CallGeneratorFamily(self.model, n_strikes=30, N_samples=50_000, rng_seed=10)

    def test_sample_matrix_shape(self):
        """Sample matrix must be (n_strikes, N_samples)."""
        assert self.fam.sample_matrix.shape == (30, 50_000)

    def test_centering_all_ok(self):
        """All generators must be centered within tolerance."""
        df = self.fam.verify_centering(tol=5e-3)
        assert df["centered_ok"].all(), (
            f"Not all generators centered: max error = {df['empirical_mean'].abs().max():.6f}"
        )

    def test_l0_boundedness_table_columns(self):
        """Boundedness table must have correct columns."""
        df = self.fam.l0_boundedness_diagnostics()
        assert "threshold_n" in df.columns
        assert "sup_P_abs_h_ge_n" in df.columns

    def test_l0_tail_decays(self):
        """L^0 tail probability must decrease with threshold for call generators."""
        df = self.fam.l0_boundedness_diagnostics(thresholds=[0.5, 1.0, 2.0, 5.0, 10.0])
        vals = df["sup_P_abs_h_ge_n"].values
        assert vals[-1] < vals[0], "Tail probability must decrease with threshold"

    def test_D_positive_and_finite(self):
        """D = sup E_P[(X_1-K)^+] must be positive and finite."""
        D = self.fam.analytical_bound_D()
        assert D > 0
        assert np.isfinite(D)


class TestAssumption33AndNonredundancy:
    """Tests for Assumption 3.3 and nonredundancy checks."""

    def test_assumption_33_satisfied(self):
        """Assumption 3.3 must be satisfied for the call-cone model."""
        checker = Assumption33Checker()
        result = checker.check()
        assert result["assumption_3_3"] == "satisfied"

    def test_assumption_33_reasoning_nonempty(self):
        """Reasoning must be nonempty."""
        checker = Assumption33Checker()
        result = checker.check()
        assert len(result["reasoning"]) > 20

    def test_nonredundancy_holds(self):
        """Nonredundancy K_0 ∩ L_1 = {0} must hold."""
        model = CallConeModelSpec()
        checker = NonredundancyChecker()
        result = checker.check(model)
        assert result["K0_cap_L1_eq_zero"] is True
        assert result["nonredundancy"] is True


class TestPricingMeasureFeasibility:
    """Tests for pricing measure verification."""

    def setup_method(self):
        self.model = CallConeModelSpec()
        self.pm = PricingMeasureFeasibility(self.model)

    def test_P_is_martingale_measure(self):
        """P must be an equivalent martingale measure (E_P[X_1] = X_0)."""
        result = self.pm.verify_P_in_M()
        assert result["P_is_martingale_measure"] is True

    def test_E_P_X1_equals_X0(self):
        """E_P[X_1] must equal X_0 (= c) by construction."""
        result = self.pm.verify_P_in_M()
        assert np.isclose(result["E_P_X1"], result["X0"], rtol=1e-6)

    def test_generators_centered_under_P(self):
        """E_P[h_K] = 0 for centered calls."""
        result = self.pm.verify_P_in_M()
        assert result["E_P_h_K_flat"] == 0.0
        assert result["E_P_h_K_sharp"] == 0.0

    def test_P_in_M_infty_Y(self):
        """P must be in M_infty(Y)."""
        result = self.pm.verify_P_in_M()
        assert result["P_in_M_infty_Y"] is True

    def test_M_Y_nonempty(self):
        """M(Y) must be nonempty."""
        result = self.pm.verify_P_in_M()
        assert result["M_Y_nonempty"] is True

    def test_D_finite(self):
        """D must be finite."""
        result = self.pm.verify_P_in_M()
        assert result["D_is_finite"] is True
        assert result["D_sup_call_price"] > 0


class TestClosednessReport:
    """Tests for the closedness conclusion."""

    def test_closedness_holds(self):
        """K_0 + Y - L^0_+ must be closed."""
        result = ClosednessReport.report()
        assert result["K0_plus_Y_minus_L0plus_closed"] is True

    def test_justification_nonempty(self):
        """Justification must be nonempty."""
        result = ClosednessReport.report()
        assert len(result["justification"]) > 20


class TestRunExperiment:
    """Integration tests for the full Exp 3 pipeline."""

    def test_run_experiment_returns_dict(self):
        """run_experiment must return a dict."""
        result = run_experiment(N_samples=20_000)
        assert isinstance(result, dict)

    def test_small_cone_valid(self):
        """Small cone L_1 must be valid."""
        result = run_experiment(N_samples=20_000)
        assert result["small_cone_l1_valid"] is True

    def test_centering_ok(self):
        """All generators must be centered."""
        result = run_experiment(N_samples=20_000)
        assert result["centering_ok"] is True

    def test_M_Y_nonempty(self):
        """M(Y) must be nonempty."""
        result = run_experiment(N_samples=20_000)
        assert result["pricing_measure"]["M_Y_nonempty"] is True

    def test_closedness(self):
        """Closedness must hold."""
        result = run_experiment(N_samples=20_000)
        assert result["closedness"]["K0_plus_Y_minus_L0plus_closed"] is True
