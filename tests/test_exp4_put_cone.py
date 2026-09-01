"""
Tests for Experiment 4: Put-Option Small Cone and FTAP Validation.

Tests verify:
  - Lognormal model with put pricing
  - Put-call parity consistency
  - Centered put payoff properties (E_P[h_K]=0)
  - Lower bound q > 0 condition
  - Small-cone L^0 and L^inf boundedness
  - Assumption 3.3 and nonredundancy
  - Pricing measure verification
  - Theorem 3.5 conclusions
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp4.put_cone import (
    PutConeModelSpec,
    LowerBoundQChecker,
    PutGeneratorFamily,
    Assumption33Checker,
    NonredundancyChecker,
    PricingMeasureFeasibility,
    ClosednessReport,
    run_experiment,
)


class TestPutConeModelSpec:
    """Tests for the lognormal put-cone model."""

    def test_default_parameters(self):
        """Default model must have correct parameters."""
        m = PutConeModelSpec()
        assert m.c == 1.0
        assert m.sigma == 0.3
        assert m.K_flat == 0.8
        assert m.K_sharp == 1.2

    def test_invalid_parameters_raise(self):
        """Invalid parameters must raise AssertionError."""
        with pytest.raises(AssertionError):
            PutConeModelSpec(c=0.0)
        with pytest.raises(AssertionError):
            PutConeModelSpec(sigma=0.0)
        with pytest.raises(AssertionError):
            PutConeModelSpec(K_flat=1.5, K_sharp=1.0)

    def test_sample_x1_positive(self):
        """All lognormal samples must be positive."""
        m = PutConeModelSpec()
        samples = m.sample_x1(5000)
        assert np.all(samples > 0)

    def test_sample_x1_mean_close_to_c(self):
        """E_P[X_1] must be approximately c."""
        m = PutConeModelSpec(c=1.0, sigma=0.2)
        rng = np.random.default_rng(42)
        samples = m.sample_x1(200_000, rng=rng)
        assert abs(np.mean(samples) - m.c) < 0.01

    def test_put_price_nonnegative(self):
        """Put prices E_P[(K-X_1)^+] must be nonneg."""
        m = PutConeModelSpec()
        for K in [0.5, 0.8, 1.0, 1.2, 1.5]:
            price = m.put_price_analytical(K)
            assert price >= 0.0

    def test_put_price_increasing_in_K(self):
        """Put price must be increasing in K."""
        m = PutConeModelSpec()
        strikes = [0.5, 0.7, 0.9, 1.0, 1.2]
        prices = [m.put_price_analytical(K) for K in strikes]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1], "Put price must increase with K"

    def test_put_call_parity(self):
        """Put-call parity: C(K) - P(K) = c - K (since E_P[X_1] = c, no discounting)."""
        from src.exp3.call_cone import CallConeModelSpec
        c, sigma = 1.0, 0.3
        m_put = PutConeModelSpec(c=c, sigma=sigma)
        m_call = CallConeModelSpec(c=c, sigma=sigma)
        for K in [0.8, 1.0, 1.2]:
            put_p = m_put.put_price_analytical(K)
            call_p = m_call.call_price_analytical(K)
            # Forward put-call parity: C - P = c - K
            parity_lhs = call_p - put_p
            parity_rhs = c - K
            assert abs(parity_lhs - parity_rhs) < 1e-6, (
                f"Put-call parity violated at K={K}: C-P={parity_lhs:.6f}, c-K={parity_rhs:.6f}"
            )

    def test_centered_put_mean_near_zero(self):
        """Centered put payoff must have near-zero empirical mean."""
        m = PutConeModelSpec()
        rng = np.random.default_rng(20)
        x1 = m.sample_x1(200_000, rng=rng)
        for K in [0.8, 1.0, 1.2]:
            h_K = m.centered_put(K, x1)
            mean_h = np.mean(h_K)
            assert abs(mean_h) < 0.01, (
                f"|E[h_K]| = {abs(mean_h):.4f} at K={K} — should be ~0"
            )

    def test_put_bounded_above(self):
        """Put payoff (K-X_1)^+ <= K for all X_1 >= 0."""
        m = PutConeModelSpec()
        rng = np.random.default_rng(20)
        x1 = m.sample_x1(50_000, rng=rng)
        for K in [0.8, 1.0, 1.2]:
            raw = np.maximum(K - x1, 0.0)
            assert np.all(raw <= K + 1e-12), f"Put payoff exceeds strike K={K}"


class TestLowerBoundQChecker:
    """Tests for the q > 0 lower bound condition."""

    def setup_method(self):
        self.model = PutConeModelSpec()
        self.checker = LowerBoundQChecker(self.model, n_grid=100)

    def test_q_strictly_positive(self):
        """q = inf E_P[(K-X_1)^+] must be strictly positive."""
        q = self.checker.compute_q()
        assert q > 0.0, f"q = {q} must be > 0"

    def test_q_check_result_dict(self):
        """check() must return a dict with q_positive=True."""
        result = self.checker.check()
        assert result["q_positive"] is True

    def test_q_at_flat_positive(self):
        """Put price at K_flat must be positive."""
        result = self.checker.check()
        assert result["q_at_K_flat"] > 0.0

    def test_q_at_sharp_positive(self):
        """Put price at K_sharp must be positive."""
        result = self.checker.check()
        assert result["q_at_K_sharp"] > 0.0

    def test_q_less_than_K_flat(self):
        """q must be less than K_flat (put price bounded by strike)."""
        q = self.checker.compute_q()
        assert q < self.model.K_flat

    def test_q_varies_with_sigma(self):
        """Higher sigma => higher put prices (for ATM strikes, up to a point)."""
        m1 = PutConeModelSpec(sigma=0.1)
        m2 = PutConeModelSpec(sigma=0.4)
        q1 = LowerBoundQChecker(m1).compute_q()
        q2 = LowerBoundQChecker(m2).compute_q()
        # Higher vol => higher put price
        assert q2 > q1, f"Expected q(sigma=0.4)={q2:.4f} > q(sigma=0.1)={q1:.4f}"


class TestPutGeneratorFamily:
    """Tests for the centered put generator family H_2."""

    def setup_method(self):
        self.model = PutConeModelSpec()
        self.fam = PutGeneratorFamily(self.model, n_strikes=30, N_samples=50_000, rng_seed=20)

    def test_sample_matrix_shape(self):
        """Sample matrix must be (n_strikes, N_samples)."""
        assert self.fam.sample_matrix.shape == (30, 50_000)

    def test_centering_ok(self):
        """All put generators must be centered within tolerance."""
        df = self.fam.verify_centering(tol=5e-3)
        assert df["centered_ok"].all(), (
            f"Not all put generators centered: max error = {df['empirical_mean'].abs().max():.6f}"
        )

    def test_l_inf_bound_correct(self):
        """L^inf bound should be 2 * K_sharp."""
        bound = self.fam.l_inf_bound()
        assert np.isclose(bound, 2 * self.model.K_sharp)

    def test_sample_values_within_l_inf_bound(self):
        """All sample values must lie within the L^inf bound (analytically exact)."""
        l_inf = self.fam.l_inf_bound()
        # centered put payoff = (K-X_1)^+ - price, so max is K + price <= 2K_sharp
        assert np.all(self.fam.sample_matrix <= l_inf + 0.01)

    def test_l0_tail_decays(self):
        """Tail probability must decrease with threshold."""
        df = self.fam.l0_boundedness_diagnostics(thresholds=[0.5, 1.0, 2.0, 5.0])
        vals = df["sup_P_abs_h_ge_n"].values
        assert vals[-1] < vals[0]

    def test_D_prime_positive(self):
        """D' = sup E_P[(K-X_1)^+] must be positive."""
        D_prime = self.fam.analytical_D_prime()
        assert D_prime > 0


class TestAssumption33AndNonredundancy:
    """Tests for Assumption 3.3 and nonredundancy in put setting."""

    def test_assumption_33_satisfied(self):
        """Assumption 3.3 must be satisfied."""
        checker = Assumption33Checker()
        result = checker.check()
        assert result["assumption_3_3"] == "satisfied"

    def test_nonredundancy_holds(self):
        """K_0 ∩ L_2 = {0} must hold."""
        model = PutConeModelSpec()
        checker = NonredundancyChecker()
        result = checker.check(model)
        assert result["K0_cap_L2_eq_zero"] is True
        assert result["nonredundancy"] is True


class TestPricingMeasurePut:
    """Tests for pricing measure verification in the put setting."""

    def setup_method(self):
        self.model = PutConeModelSpec()
        self.pm = PricingMeasureFeasibility(self.model)

    def test_P_is_martingale(self):
        """P must be a martingale measure."""
        result = self.pm.verify_P_in_M()
        assert result["P_is_martingale_measure"] is True

    def test_P_in_M_infty(self):
        """P must be in M_infty(Y)."""
        result = self.pm.verify_P_in_M()
        assert result["P_in_M_infty_Y"] is True

    def test_M_Y_nonempty(self):
        """M(Y) must be nonempty."""
        result = self.pm.verify_P_in_M()
        assert result["M_Y_nonempty"] is True


class TestRunExperiment:
    """Integration tests for the full Exp 4 pipeline."""

    def test_run_experiment_returns_dict(self):
        result = run_experiment(N_samples=20_000)
        assert isinstance(result, dict)

    def test_q_positive(self):
        """q > 0 must be confirmed."""
        result = run_experiment(N_samples=20_000)
        assert result["q_condition"]["q_positive"] is True

    def test_small_cone_valid(self):
        result = run_experiment(N_samples=20_000)
        assert result["small_cone_l2_valid"] is True

    def test_centering_ok(self):
        result = run_experiment(N_samples=20_000)
        assert result["centering_ok"] is True

    def test_M_Y_nonempty(self):
        result = run_experiment(N_samples=20_000)
        assert result["pricing_measure"]["M_Y_nonempty"] is True

    def test_closedness(self):
        result = run_experiment(N_samples=20_000)
        assert result["closedness"]["K0_plus_Y_minus_L0plus_closed"] is True
