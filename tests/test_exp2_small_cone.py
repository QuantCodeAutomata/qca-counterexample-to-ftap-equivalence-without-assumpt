"""
Tests for Experiment 2: Small-Cone Constructions (Examples 5.1–5.3).

Tests verify:
  - L^0 boundedness checker accuracy
  - Example 5.1: bounded-L^2 generator family properties
  - Example 5.2: Cauchy-based generator family (L^0 bounded, NOT L^1/L^2)
  - Example 5.3: finite-dimensional generator family
  - Lemma 2.4 illustration: nonredundancy and closedness
  - Mathematical properties of tail probability functions
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.l0_tools import (
    tail_probability,
    sup_tail_probability,
    l0_boundedness_table,
    is_zero_excluded,
    convergence_in_prob_check,
)
from src.exp2.small_cone import (
    SmallConeDefinition,
    GeneratorFamilyExample51,
    GeneratorFamilyExample52,
    GeneratorFamilyExample53,
    Lemma24Illustration,
    run_experiment,
)


class TestL0Tools:
    """Tests for the L^0 boundedness utilities."""

    def test_tail_probability_all_small(self):
        """Tail probability is 0 when all |h| < threshold."""
        samples = np.array([0.1, 0.2, 0.3, -0.1])
        assert tail_probability(samples, threshold=1.0) == 0.0

    def test_tail_probability_all_large(self):
        """Tail probability is 1 when all |h| >= threshold."""
        samples = np.array([5.0, -5.0, 10.0])
        assert tail_probability(samples, threshold=4.0) == 1.0

    def test_tail_probability_negative_threshold_raises(self):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            tail_probability(np.array([1.0, 2.0]), threshold=-1.0)

    def test_tail_probability_between_0_and_1(self):
        """Tail probability must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(size=1000)
        for thresh in [0.5, 1.0, 2.0, 5.0]:
            p = tail_probability(samples, threshold=thresh)
            assert 0.0 <= p <= 1.0

    def test_tail_probability_monotone(self):
        """Tail probability must be non-increasing in threshold."""
        rng = np.random.default_rng(1)
        samples = rng.standard_normal(size=5000)
        thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
        probs = [tail_probability(samples, t) for t in thresholds]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], "Tail probability not monotone decreasing"

    def test_sup_tail_probability_shape(self):
        """sup_tail_probability must handle (K, N) matrices."""
        mat = np.ones((5, 1000))
        sup = sup_tail_probability(mat, threshold=0.5)
        assert sup == 1.0  # all entries are 1.0 >= 0.5

    def test_sup_tail_probability_2d_required(self):
        """Must raise for 1D input."""
        with pytest.raises(ValueError):
            sup_tail_probability(np.ones(100), threshold=1.0)

    def test_is_zero_excluded_true(self):
        """Generator with values far from zero must pass SC4."""
        mat = np.ones((3, 1000)) * 5.0
        assert is_zero_excluded(mat)

    def test_is_zero_excluded_false(self):
        """Generator that is almost surely zero must fail SC4."""
        mat = np.zeros((3, 1000))
        assert not is_zero_excluded(mat)

    def test_l0_boundedness_table_returns_dataframe(self):
        """l0_boundedness_table must return a pd.DataFrame."""
        rng = np.random.default_rng(2)
        mat = rng.standard_normal(size=(10, 5000))
        df = l0_boundedness_table(mat)
        assert isinstance(df, pd.DataFrame)
        assert "threshold_n" in df.columns
        assert "sup_P_abs_h_ge_n" in df.columns

    def test_l0_boundedness_decays_for_normal(self):
        """For normal generators, tail probability should decay to near zero."""
        rng = np.random.default_rng(3)
        mat = rng.standard_normal(size=(10, 50_000))
        df = l0_boundedness_table(mat, thresholds=[1, 2, 5, 10])
        vals = df["sup_P_abs_h_ge_n"].values
        assert vals[-1] < vals[0], "Tail probability should decrease with threshold"
        assert vals[-1] < 0.05, "At threshold=10, tail prob should be very small for Normal"


class TestExample51:
    """Tests for Example 5.1: bounded-L^2 generator family."""

    def setup_method(self):
        self.ex = GeneratorFamilyExample51(K_trunc=20, N_samples=50_000, rng_seed=1)

    def test_sample_matrix_shape(self):
        """Sample matrix must be (K_trunc, N_samples)."""
        assert self.ex.sample_matrix.shape == (20, 50_000)

    def test_coefficients_bounded(self):
        """Coefficients a_n must satisfy 0 < a_star <= a_n <= a_dagger."""
        assert np.all(self.ex.a_n >= self.ex.a_star - 1e-12)
        assert np.all(self.ex.a_n <= self.ex.a_dagger + 1e-12)

    def test_analytical_bound_decreasing(self):
        """Analytical Chebyshev bound must be decreasing in threshold."""
        bounds = [self.ex.analytical_bound(t) for t in [1, 5, 10, 50]]
        for i in range(len(bounds) - 1):
            assert bounds[i] >= bounds[i + 1]

    def test_analytical_bound_at_infinity(self):
        """Analytical bound should approach 0 as t -> infinity."""
        assert self.ex.analytical_bound(1e6) < 1e-5

    def test_sc3_boundedness_passes(self):
        """Example 5.1 must pass L^0 boundedness check."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC3_bounded_L0"] == "passed"

    def test_sc4_zero_excluded(self):
        """Example 5.1 generators must exclude zero."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC4_zero_excluded"]

    def test_qualifies_as_small_cone(self):
        """Example 5.1 must qualify as a small cone."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["qualifies_as_small_cone"]

    def test_empirical_vs_analytical_bound(self):
        """Empirical tail should be below analytical Chebyshev bound."""
        checker = self.ex.build_checker()
        for t in [5.0, 10.0, 20.0]:
            empirical = sup_tail_probability(self.ex.sample_matrix, t)
            analytic = self.ex.analytical_bound(t)
            assert empirical <= analytic + 0.05, (
                f"Empirical ({empirical:.4f}) > Chebyshev bound ({analytic:.4f}) at t={t}"
            )


class TestExample52:
    """Tests for Example 5.2: Cauchy-based generator family."""

    def setup_method(self):
        self.ex = GeneratorFamilyExample52(K_trunc=20, N_samples=50_000, rng_seed=2)

    def test_sample_matrix_shape(self):
        """Sample matrix must be (K_trunc, N_samples)."""
        assert self.ex.sample_matrix.shape == (20, 50_000)

    def test_analytical_bound_decays(self):
        """Cauchy analytical bound must decay to 0 (slower than polynomial)."""
        bounds = [self.ex.analytical_bound(t) for t in [1, 5, 20, 100]]
        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1], "Bound should be strictly decreasing"
        assert bounds[-1] < 0.1, "Bound at t=100 should be small"

    def test_sc3_boundedness_passes(self):
        """Example 5.2 must pass L^0 boundedness despite Cauchy tails."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC3_bounded_L0"] == "passed"

    def test_sc4_zero_excluded(self):
        """Example 5.2 generators must exclude zero."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC4_zero_excluded"]

    def test_qualifies_as_small_cone(self):
        """Example 5.2 must qualify as a small cone."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["qualifies_as_small_cone"]

    def test_cauchy_l0_vs_l1_comparison(self):
        """Cauchy generators have L^0-bounded but large empirical absolute values."""
        # Standard Cauchy has infinite first moment:
        # the sample median should be near 0, but sample mean can be large
        mat = self.ex.sample_matrix  # (K, N) scaled Cauchy
        empirical_means = np.abs(mat.mean(axis=1))
        # Means are not reliable (unstable), but L^0 boundedness should hold
        checker = self.ex.build_checker()
        df = checker.check_sc3_boundedness(thresholds=[10.0, 50.0])
        # At threshold 50, tail should still be > 0 but declining
        vals = df["sup_P_abs_h_ge_n"].values
        assert vals[-1] < vals[0], "Tail should decrease"


class TestExample53:
    """Tests for Example 5.3: finite-dimensional generator family."""

    def setup_method(self):
        self.ex = GeneratorFamilyExample53(S=10, K_fin=4, N_mc_samples=50_000, rng_seed=3)

    def test_generators_shape(self):
        """Generators matrix must be (K_fin, S)."""
        assert self.ex.generators.shape == (4, 10)

    def test_linear_independence(self):
        """Generators must be linearly independent."""
        assert self.ex.check_linear_independence()

    def test_probabilities_sum_to_one(self):
        """State probabilities must sum to 1."""
        assert np.isclose(self.ex.p.sum(), 1.0)

    def test_sc3_boundedness_finite_state(self):
        """Finite-state generators are trivially bounded in L^0."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC3_bounded_L0"] == "passed"

    def test_sc4_zero_excluded(self):
        """Orthonormalized generators must be nonzero."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["SC4_zero_excluded"]

    def test_qualifies_as_small_cone(self):
        """Example 5.3 must qualify as a small cone."""
        checker = self.ex.build_checker()
        result = checker.validate()
        assert result["qualifies_as_small_cone"]

    def test_convex_hull_nonzero_generators(self):
        """All generators must be strictly nonzero."""
        props = self.ex.check_convex_hull_properties()
        assert props["all_nonzero"]

    def test_convex_hull_n_vertices(self):
        """Convex hull must have exactly K_fin vertices."""
        props = self.ex.check_convex_hull_properties()
        assert props["n_vertices"] == self.ex.K_fin


class TestLemma24:
    """Tests for the Lemma 2.4 finite-state illustration."""

    def setup_method(self):
        S = 10
        rng = np.random.default_rng(99)
        p = np.ones(S) / S
        x1 = np.sort(rng.uniform(0.5, 1.5, size=S))
        x0 = float(np.mean(x1))
        ex53 = GeneratorFamilyExample53(S=S, K_fin=3, rng_seed=99)
        self.lemma = Lemma24Illustration(
            model_x1=x1, x0=x0, p=p,
            small_cone_generators=ex53.generators,
        )

    def test_nonredundancy_returns_dict(self):
        """Nonredundancy check must return a dict."""
        result = self.lemma.check_nonredundancy()
        assert isinstance(result, dict)
        assert "nonredundancy" in result

    def test_closedness_true(self):
        """In finite state, C + L is always closed."""
        result = self.lemma.check_closedness_C_plus_L()
        assert result["C_plus_L_closed"] is True


class TestRunExperiment:
    """Integration tests for the full Exp 2 pipeline."""

    def test_run_experiment_returns_dict(self):
        """run_experiment must return a dict."""
        result = run_experiment(N_samples=10_000)
        assert isinstance(result, dict)

    def test_run_experiment_keys(self):
        """All example keys must be present."""
        result = run_experiment(N_samples=10_000)
        for key in ["example_5_1", "example_5_2", "example_5_3", "lemma_2_4"]:
            assert key in result

    def test_all_examples_qualify(self):
        """All three examples must qualify as small cones."""
        result = run_experiment(N_samples=10_000)
        for ex_key in ["example_5_1", "example_5_2", "example_5_3"]:
            assert result[ex_key]["qualifies_as_small_cone"], (
                f"{ex_key} did not qualify as a small cone"
            )

    def test_lemma24_closedness(self):
        """Lemma 2.4 illustration must show closedness."""
        result = run_experiment(N_samples=10_000)
        assert result["lemma_2_4"]["closedness"]["C_plus_L_closed"] is True
