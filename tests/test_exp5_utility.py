"""
Tests for Experiment 5: Existence of Optimal Semi-Static Strategy (Theorem 4.1).

Tests verify:
  - Utility function mathematical properties (monotone, concave, bounded above)
  - Finite-state model correctness
  - Static set generator properties
  - No-arbitrage verification
  - Pricing measure construction
  - Optimizer behavior and correctness
  - Optimal payoff decomposition
  - Maximizing sequence stabilization
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp5.utility_opt import (
    cara_utility,
    arctan_utility,
    UtilityModelSpec,
    SmallConeStaticSet,
    NoArbitrageChecker,
    PricingMeasureConstructor,
    UtilityOptimizer,
    OptimizerDiagnostics,
    run_experiment,
)
from src.common.finite_state import FiniteStateModel


class TestUtilityFunctions:
    """Tests for bounded-above concave utility functions."""

    def test_cara_increasing(self):
        """CARA utility must be strictly increasing."""
        x = np.linspace(-5, 5, 100)
        u = cara_utility(x, alpha=1.0)
        diffs = np.diff(u)
        assert np.all(diffs > 0), "CARA utility must be strictly increasing"

    def test_cara_concave(self):
        """CARA utility must be strictly concave (second differences negative)."""
        x = np.linspace(-3, 3, 100)
        u = cara_utility(x, alpha=1.0)
        second_diff = np.diff(u, n=2)
        assert np.all(second_diff < 0), "CARA utility must be strictly concave"

    def test_cara_bounded_above(self):
        """CARA utility must be bounded above by 1.

        We test at x values small enough that float64 can represent
        1 - exp(-x) < 1 exactly (requires exp(-x) > machine_epsilon ≈ 2.2e-16,
        i.e. x < ~36 with alpha=1).  At very large x the subtraction 1 - eps
        rounds to exactly 1.0 in float64; that is a precision artefact, not a
        violation of the mathematical bound.
        """
        x = np.array([0.0, 5.0, 10.0, 30.0])
        u = cara_utility(x, alpha=1.0)
        assert np.all(u < 1.0), "CARA utility must be < 1 for representable x values"

    def test_cara_upper_bound_approached(self):
        """CARA utility must approach 1 as x -> infinity."""
        large_x = np.array([1e6])
        assert np.isclose(cara_utility(large_x, alpha=1.0)[0], 1.0, atol=1e-5)

    def test_cara_alpha_affects_risk_aversion(self):
        """Higher alpha => more risk-averse: saturates to 1 faster for x>0.

        For u(x; alpha) = 1 - exp(-alpha*x) and x > 0:
          d/d(alpha) u(x; alpha) = x * exp(-alpha*x) > 0
        so a larger alpha always gives a larger utility value for the same
        positive x (the function reaches its upper bound faster).
        We also check that the absolute curvature |u''(0)| = alpha^2 increases
        with alpha, confirming higher risk aversion.
        """
        x = np.linspace(0.01, 2.0, 100)
        u1 = cara_utility(x, alpha=0.5)
        u2 = cara_utility(x, alpha=2.0)
        # Higher alpha → utility saturates faster → larger value at each x > 0.
        assert u2[-1] > u1[-1], "Higher alpha should give higher utility for positive x"
        # Absolute curvature |u''(0)| = alpha^2 increases with alpha.
        dx = x[1] - x[0]
        curv1 = abs(np.gradient(np.gradient(u1, dx), dx)[len(x) // 2])
        curv2 = abs(np.gradient(np.gradient(u2, dx), dx)[len(x) // 2])
        assert curv2 > curv1, "Higher alpha must give larger absolute curvature"

    def test_arctan_increasing(self):
        """Arctan utility must be strictly increasing."""
        x = np.linspace(-5, 5, 100)
        u = arctan_utility(x)
        diffs = np.diff(u)
        assert np.all(diffs > 0), "Arctan utility must be strictly increasing"

    def test_arctan_bounded_above(self):
        """Arctan utility must be bounded above by pi/2."""
        x = np.array([10.0, 100.0, 1e6])
        u = arctan_utility(x)
        assert np.all(u < np.pi / 2), "Arctan utility must be < pi/2"

    def test_arctan_concave_for_positive_x(self):
        """Arctan utility is concave for x > 0."""
        x = np.linspace(0.01, 5, 100)
        u = arctan_utility(x)
        second_diff = np.diff(u, n=2)
        assert np.all(second_diff < 0), "Arctan must be concave for x > 0"

    def test_cara_value_at_zero(self):
        """CARA(0) = 1 - exp(0) = 0."""
        assert np.isclose(cara_utility(np.array([0.0]))[0], 0.0)

    def test_arctan_value_at_zero(self):
        """Arctan(0) = 0."""
        assert np.isclose(arctan_utility(np.array([0.0]))[0], 0.0)


class TestUtilityModelSpec:
    """Tests for the finite-state model specification."""

    def test_probabilities_sum_to_one(self):
        """State probabilities must sum to 1."""
        m = UtilityModelSpec(S=8)
        assert np.isclose(m.p.sum(), 1.0)

    def test_all_probabilities_positive(self):
        """All state probabilities must be strictly positive."""
        m = UtilityModelSpec(S=8)
        assert np.all(m.p > 0)

    def test_x0_equals_E_p_x1(self):
        """X_0 must equal E_P[X_1] (martingale under P)."""
        m = UtilityModelSpec(S=8, rng_seed=42)
        expected_x1 = float(np.dot(m.p, m.x1))
        assert np.isclose(m.x0, expected_x1, rtol=1e-10)

    def test_x1_positive(self):
        """Asset terminal values must be positive."""
        m = UtilityModelSpec(S=8)
        assert np.all(m.x1 > 0)

    def test_x1_sorted(self):
        """X_1 values are sorted."""
        m = UtilityModelSpec(S=8)
        assert np.all(np.diff(m.x1) >= 0)

    def test_finite_state_model_returns_fsm(self):
        """build_finite_state_model must return a FiniteStateModel instance."""
        m = UtilityModelSpec(S=8)
        fsm = m.build_finite_state_model()
        assert isinstance(fsm, FiniteStateModel)


class TestSmallConeStaticSet:
    """Tests for the finite-generator small-cone static set."""

    def setup_method(self):
        self.model = UtilityModelSpec(S=8, rng_seed=42)
        self.sset = SmallConeStaticSet(self.model, K_gen=3)

    def test_generators_shape(self):
        """Generator matrix must be (K_gen, S)."""
        gens = self.sset.get_generators()
        assert gens.shape == (3, 8)

    def test_generators_nonzero(self):
        """All generators must be nonzero."""
        gens = self.sset.get_generators()
        norms = np.linalg.norm(gens, axis=1)
        assert np.all(norms > 1e-10)

    def test_generators_centered_under_P(self):
        """Each generator must be centered under P (E_P[g_k] = 0)."""
        p = self.model.p
        gens = self.sset.get_generators()
        for k, g in enumerate(gens):
            e_g = np.dot(p, g)
            assert abs(e_g) < 1e-8, f"Generator {k} not centered: E_P[g_{k}] = {e_g:.6e}"

    def test_payoff_nonneg_lambdas(self):
        """Payoff function must accept nonneg lambdas."""
        lambdas = np.array([0.5, 1.0, 0.0])
        h = self.sset.payoff(lambdas)
        assert h.shape == (8,)

    def test_payoff_zero_lambdas(self):
        """Zero lambdas must give zero payoff."""
        lambdas = np.zeros(3)
        h = self.sset.payoff(lambdas)
        np.testing.assert_allclose(h, 0.0)

    def test_payoff_linearity_in_lambdas(self):
        """Payoff must be linear in lambdas."""
        l1 = np.array([1.0, 0.0, 0.0])
        l2 = np.array([0.0, 1.0, 0.0])
        h_sum = self.sset.payoff(l1 + l2)
        h1_plus_h2 = self.sset.payoff(l1) + self.sset.payoff(l2)
        np.testing.assert_allclose(h_sum, h1_plus_h2, atol=1e-12)


class TestNoArbitrageChecker:
    """Tests for the NA(Y) check in the utility model."""

    def test_dynamic_na(self):
        """Dynamic NA must hold (X_1 is P-martingale with martingale measure P)."""
        model = UtilityModelSpec(S=8, rng_seed=42)
        fsm = model.build_finite_state_model()
        sset = SmallConeStaticSet(model, K_gen=3)
        checker = NoArbitrageChecker(fsm, sset)
        result = checker.check()
        assert result["dynamic_na"] is True

    def test_semi_static_na(self):
        """Semi-static NA(Y) must hold."""
        model = UtilityModelSpec(S=8, rng_seed=42)
        fsm = model.build_finite_state_model()
        sset = SmallConeStaticSet(model, K_gen=3)
        checker = NoArbitrageChecker(fsm, sset)
        result = checker.check()
        assert result["semi_static_na"] is True


class TestPricingMeasureConstructor:
    """Tests for the pricing measure (P) in the utility model."""

    def setup_method(self):
        self.model = UtilityModelSpec(S=8, rng_seed=42)
        self.fsm = self.model.build_finite_state_model()
        self.sset = SmallConeStaticSet(self.model, K_gen=3)
        self.pmc = PricingMeasureConstructor(self.fsm, self.sset)

    def test_P_martingale(self):
        """P must be a martingale measure."""
        result = self.pmc.construct()
        assert result["P_martingale"] is True

    def test_generators_all_centered(self):
        """All generators must be centered under P."""
        result = self.pmc.construct()
        assert result["all_generators_centered"] is True

    def test_P_in_M_infty_Y(self):
        """P must be in M_infty(Y)."""
        result = self.pmc.construct()
        assert result["P_in_M_infty_Y"] is True

    def test_E_P_X1_close_to_X0(self):
        """E_P[X_1] must equal X_0."""
        result = self.pmc.construct()
        assert np.isclose(result["E_P_X1"], result["X0"], rtol=1e-10)


class TestUtilityOptimizer:
    """Tests for the utility maximization solver."""

    def setup_method(self):
        self.model = UtilityModelSpec(S=8, rng_seed=42)
        self.fsm = self.model.build_finite_state_model()
        self.sset = SmallConeStaticSet(self.model, K_gen=3)

    def _make_optimizer(self, utility_name="CARA"):
        if utility_name == "CARA":
            return UtilityOptimizer(
                self.fsm, self.sset,
                utility_fn=lambda x: cara_utility(x, alpha=1.0),
                utility_name="CARA u(x)=1-exp(-x)",
            )
        else:
            return UtilityOptimizer(
                self.fsm, self.sset,
                utility_fn=arctan_utility,
                utility_name="Arctan u(x)=arctan(x)",
            )

    def test_terminal_wealth_shape(self):
        """Terminal wealth must have shape (S,)."""
        opt = self._make_optimizer()
        params = np.concatenate([[0.5], [1.0, 0.0, 0.5]])
        vt = opt._terminal_wealth(params)
        assert vt.shape == (8,)

    def test_terminal_wealth_composition(self):
        """Terminal wealth = dynamic + static."""
        opt = self._make_optimizer()
        phi = 1.5
        lambdas = np.array([1.0, 0.0, 0.0])
        params = np.concatenate([[phi], lambdas])
        vt = opt._terminal_wealth(params)
        dynamic = phi * opt.gain
        static = opt.generators[0]  # only first generator with weight 1
        expected = dynamic + static
        np.testing.assert_allclose(vt, expected, atol=1e-12)

    def test_neg_expected_utility_scalar(self):
        """Negative expected utility must return a scalar."""
        opt = self._make_optimizer()
        params = np.zeros(1 + self.sset.K_gen)
        val = opt._neg_expected_utility(params)
        assert np.isscalar(val)

    def test_optimizer_cara_decreases_with_more_static(self):
        """With more static budget, utility should not decrease."""
        opt = self._make_optimizer("CARA")
        # Zero static position
        p0 = np.zeros(1 + self.sset.K_gen)
        eu0 = -opt._neg_expected_utility(p0)

        # Run optimization
        result = opt.optimize(n_restarts=5)
        eu_opt = result["optimal_expected_utility"]
        assert eu_opt >= eu0, (
            f"Optimizer should find EU >= baseline: {eu_opt:.4f} vs baseline {eu0:.4f}"
        )

    def test_optimizer_cara_runs(self):
        """CARA optimizer must run without error."""
        opt = self._make_optimizer("CARA")
        result = opt.optimize(n_restarts=5)
        assert "optimal_phi" in result
        assert "optimal_lambdas" in result
        assert "optimal_expected_utility" in result

    def test_optimizer_arctan_runs(self):
        """Arctan optimizer must run without error."""
        opt = self._make_optimizer("Arctan")
        result = opt.optimize(n_restarts=5)
        assert "optimal_expected_utility" in result

    def test_optimal_lambdas_nonneg(self):
        """Optimal lambdas must be nonneg."""
        opt = self._make_optimizer("CARA")
        result = opt.optimize(n_restarts=10)
        lambdas = np.array(result["optimal_lambdas"])
        assert np.all(lambdas >= -1e-6), f"Lambdas not nonneg: {lambdas}"

    def test_optimal_eu_within_utility_bounds(self):
        """Optimal EU must be within [u(-inf), u(+inf)] bounds."""
        opt = self._make_optimizer("CARA")
        result = opt.optimize(n_restarts=10)
        eu = result["optimal_expected_utility"]
        # CARA bounded above by 1
        assert eu < 1.0 + 1e-6, f"EU = {eu:.4f} exceeds CARA upper bound 1"

    def test_optimal_vt_shape(self):
        """Optimal V_T must have shape (S,)."""
        opt = self._make_optimizer("CARA")
        result = opt.optimize(n_restarts=5)
        assert len(result["optimal_V_T"]) == 8

    def test_payoff_decomposition_sums_correctly(self):
        """Dynamic + static parts must sum to total V_T."""
        opt = self._make_optimizer("CARA")
        result = opt.optimize(n_restarts=5)
        vt = np.array(result["optimal_V_T"])
        dyn = np.array(result["optimal_dynamic_part"])
        stat = np.array(result["optimal_static_part"])
        np.testing.assert_allclose(vt, dyn + stat, atol=1e-8)


class TestOptimizerDiagnostics:
    """Tests for the maximizing sequence illustration."""

    def test_sequence_length(self):
        """Maximizing sequence must have the requested number of steps."""
        model = UtilityModelSpec(S=8, rng_seed=42)
        fsm = model.build_finite_state_model()
        sset = SmallConeStaticSet(model, K_gen=3)
        opt = UtilityOptimizer(fsm, sset, utility_fn=lambda x: cara_utility(x, 1.0))
        diag = OptimizerDiagnostics(opt)

        params_star = np.zeros(1 + sset.K_gen)
        df = diag.maximizing_sequence(params_star, n_steps=30)
        assert len(df) == 30

    def test_sequence_columns(self):
        """Sequence DataFrame must have 'step' and 'expected_utility' columns."""
        model = UtilityModelSpec(S=8, rng_seed=42)
        fsm = model.build_finite_state_model()
        sset = SmallConeStaticSet(model, K_gen=3)
        opt = UtilityOptimizer(fsm, sset, utility_fn=lambda x: cara_utility(x, 1.0))
        diag = OptimizerDiagnostics(opt)

        params_star = np.zeros(1 + sset.K_gen)
        df = diag.maximizing_sequence(params_star, n_steps=10)
        assert "step" in df.columns
        assert "expected_utility" in df.columns

    def test_sequence_eu_within_bounds(self):
        """All EU values in the maximizing sequence must be within utility bounds."""
        model = UtilityModelSpec(S=8, rng_seed=42)
        fsm = model.build_finite_state_model()
        sset = SmallConeStaticSet(model, K_gen=3)
        opt = UtilityOptimizer(fsm, sset, utility_fn=lambda x: cara_utility(x, 1.0))
        diag = OptimizerDiagnostics(opt)

        params_star = np.zeros(1 + sset.K_gen)
        df = diag.maximizing_sequence(params_star, n_steps=20)
        # CARA is bounded above by 1, so EU < 1
        assert np.all(df["expected_utility"].values < 1.0 + 1e-6)


class TestRunExperiment:
    """Integration tests for the full Exp 5 pipeline."""

    def test_run_experiment_returns_dict(self):
        result = run_experiment()
        assert isinstance(result, dict)

    def test_no_arbitrage_passes(self):
        result = run_experiment()
        assert result["no_arbitrage"]["semi_static_na"] is True

    def test_P_in_M_infty_Y(self):
        result = run_experiment()
        assert result["pricing_measure"]["P_in_M_infty_Y"] is True

    def test_cara_optimizer_exists(self):
        result = run_experiment()
        assert "cara_optimization" in result
        assert result["cara_optimization"]["optimizer_converged"] is True

    def test_arctan_optimizer_exists(self):
        result = run_experiment()
        assert "arctan_optimization" in result
        assert result["arctan_optimization"]["optimizer_converged"] is True

    def test_theorem_conclusion_in_result(self):
        result = run_experiment()
        assert "theorem_4_1_conclusion" in result

    def test_maximizing_sequence_present(self):
        result = run_experiment()
        assert "maximizing_sequence_cara" in result
        assert len(result["maximizing_sequence_cara"]) > 0
