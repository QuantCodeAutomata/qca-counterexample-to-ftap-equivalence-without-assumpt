"""
Experiment 2 — Validation of Small-Cone Constructions (Examples 5.1–5.3).

A generator set H is a SMALL CONE generator if:
  (SC1)  H is convex.
  (SC2)  H is closed in L^0 (convergence in probability).
  (SC3)  H is bounded in L^0: sup_{h in H} P(|h| >= n) -> 0 as n -> infinity.
  (SC4)  0 is not in H.

The cone L = C(H) = { lambda * h : lambda >= 0, h in H } is then called a
small cone.  It plays the role of the static payoff set Y in the paper's FTAP.

Example 5.1 — Bounded-L^2 construction:
    Generator family H_1 = { a_n * xi_n : n >= 1 } where xi_n are independent
    square-integrable r.v.s with sup_n E[xi_n^2] <= J < inf and
    0 < a_* <= a_n <= a_dag.  The family is bounded in L^0 because finite
    second moments imply tail decay: P(|a_n xi_n| >= t) <= J a_dag^2 / t^2.

Example 5.2 — Cauchy-based construction:
    Generator family H_2 = { c_n * zeta_n : n >= 1 } where zeta_n are i.i.d.
    standard Cauchy and c_n is in a compact interval [c_*, c_dag], both > 0.
    Boundedness in L^0 follows from: P(|c_n zeta_n| >= t) = P(|zeta_n| >= t/c_n)
    <= P(|zeta_1| >= t/c_dag) -> 0 as t -> inf uniformly over n.
    NOTE: H_2 is NOT bounded in L^1 or L^2 because E[|zeta_n|] = infinity.

Example 5.3 — Finite-dimensional:
    Generator family H_3 is a finite set of linearly independent random variables
    on a finite probability space.  All small-cone properties are verifiable by
    exact linear algebra.

Custom — Context7 found no library equivalent for small-cone verification
(paper Section 2 and Examples 5.1–5.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from src.common.l0_tools import (
    l0_boundedness_table,
    sup_tail_probability,
    convergence_in_prob_check,
    is_zero_excluded,
)


# ---------------------------------------------------------------------------
# Small-cone definition checker
# ---------------------------------------------------------------------------

class SmallConeDefinition:
    """
    Checks whether a generator family H satisfies the small-cone conditions.

    The family H is represented as a sample matrix of shape (K, N) where K is
    the number of generators (or a truncated finite subset for illustration) and
    N is the Monte Carlo sample size.

    For infinite-dimensional examples, K represents a truncated finite subset
    explicitly labeled as such.

    Parameters
    ----------
    sample_matrix : np.ndarray, shape (K, N)
        Rows are generators, columns are Monte Carlo observations.
    name : str
        Name of the example.
    is_truncated : bool
        If True, K < infinity and the full family is infinite-dimensional.
    analytical_notes : str
        Analytical justification for the small-cone properties.
    """

    def __init__(
        self,
        sample_matrix: np.ndarray,
        name: str,
        is_truncated: bool = False,
        analytical_notes: str = "",
    ) -> None:
        assert sample_matrix.ndim == 2, "sample_matrix must be 2-D (K, N)"
        self.sample_matrix = sample_matrix
        self.name = name
        self.is_truncated = is_truncated
        self.analytical_notes = analytical_notes
        self.K, self.N = sample_matrix.shape

    def check_sc3_boundedness(
        self, thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Check (SC3): boundedness in L^0 via tail probability decay.

        Returns
        -------
        pd.DataFrame
            Table of sup_H P(|h| >= n) over increasing thresholds.
        """
        return l0_boundedness_table(self.sample_matrix, thresholds)

    def check_sc4_zero_excluded(self) -> bool:
        """
        Check (SC4): 0 not in H (heuristic for sampled generators).

        Returns
        -------
        bool
        """
        return is_zero_excluded(self.sample_matrix)

    def check_sc1_convexity_note(self) -> str:
        """
        Return analytical note on convexity (SC1).

        Convexity of the generator set H cannot be checked purely from
        a finite sample of individual generators without sampling their
        convex combinations.  The analytical argument is recorded here.
        """
        return (
            "SC1 (Convexity): Analytically, conv(H) is formed by taking all "
            "convex combinations of generators.  For Examples 5.1–5.2, H consists "
            "of an indexed family; its convex hull still satisfies boundedness "
            "in L^0 because tail probabilities are subadditive under convex mixing. "
            "For Example 5.3, H is finite and conv(H) is a polytope, directly convex."
        )

    def full_property_table(self) -> pd.DataFrame:
        """
        Build a summary table of small-cone property checks.

        Returns
        -------
        pd.DataFrame
        """
        sc3_df = self.check_sc3_boundedness()
        # Use tail probability at n=10 as a representative value
        sup_at_10 = sup_tail_probability(self.sample_matrix, 10.0)
        sup_at_50 = sup_tail_probability(self.sample_matrix, 50.0)
        sc4 = self.check_sc4_zero_excluded()

        records = [
            {
                "property": "SC1 — Convexity",
                "status": "Analytical — see notes",
                "value": "n/a",
            },
            {
                "property": "SC2 — Closed in L^0",
                "status": "Analytical — closed-in-prob",
                "value": "n/a",
            },
            {
                "property": "SC3 — Bounded in L^0 (sup P(|h|>=10))",
                "status": "Numerical + Analytical",
                "value": f"{sup_at_10:.4f}",
            },
            {
                "property": "SC3 — Bounded in L^0 (sup P(|h|>=50))",
                "status": "Numerical + Analytical",
                "value": f"{sup_at_50:.4f}",
            },
            {
                "property": "SC4 — Zero excluded from H",
                "status": "Passed" if sc4 else "FAILED",
                "value": str(sc4),
            },
        ]
        df = pd.DataFrame(records)
        df.attrs["name"] = self.name
        df.attrs["is_truncated"] = self.is_truncated
        return df

    def validate(self) -> dict:
        """
        Return full validation result dict.
        """
        sc3_df = self.check_sc3_boundedness()
        sc4 = self.check_sc4_zero_excluded()
        # Check L^0 boundedness: sup tail prob must decay to 0.
        # Two cases pass SC3:
        #   (a) Tail probabilities strictly decrease with threshold, OR
        #   (b) All tail probabilities are already 0 (trivially bounded, e.g. finite state).
        tail_values = sc3_df["sup_P_abs_h_ge_n"].values
        all_zero = bool(np.all(tail_values == 0.0))
        sc3_passes = bool(all_zero or (tail_values[-1] < tail_values[0]))
        return {
            "name": self.name,
            "is_truncated": self.is_truncated,
            "K_generators_shown": self.K,
            "N_samples": self.N,
            "SC1_convexity": "analytical",
            "SC2_closed_L0": "analytical",
            "SC3_bounded_L0": "passed" if sc3_passes else "check_needed",
            "SC3_sup_tail_at_10": float(sup_tail_probability(self.sample_matrix, 10.0)),
            "SC3_sup_tail_at_100": float(sup_tail_probability(self.sample_matrix, 100.0)),
            "SC4_zero_excluded": sc4,
            "qualifies_as_small_cone": sc3_passes and sc4,
            "analytical_notes": self.analytical_notes,
        }


# ---------------------------------------------------------------------------
# Example 5.1 — Bounded-L^2 generator family
# ---------------------------------------------------------------------------

class GeneratorFamilyExample51:
    """
    Example 5.1: Independent square-integrable generators with bounded coefficients.

    xi_n ~ N(0,1) i.i.d. (extra-paper choice, consistent with sup E[xi_n^2] <= 1)
    a_n in [a_star, a_dagger] with 0 < a_star <= a_dagger.

    The generator family is H_1 = { a_n * xi_n : n = 1, 2, ..., K_trunc }.

    Boundedness in L^0:
      P(|a_n * xi_n| >= t) <= P(|xi_n| >= t / a_dagger)  (by a_n <= a_dagger)
                             = 2 * Phi(-t / a_dagger)
      This goes to 0 uniformly in n as t -> inf.

    Custom — paper Example 5.1.
    """

    def __init__(
        self,
        K_trunc: int = 50,
        N_samples: int = 100_000,
        a_star: float = 0.5,
        a_dagger: float = 2.0,
        rng_seed: int = 1,
    ) -> None:
        self.K_trunc = K_trunc
        self.N_samples = N_samples
        self.a_star = a_star
        self.a_dagger = a_dagger
        self.rng_seed = rng_seed

        rng = np.random.default_rng(rng_seed)
        # Deterministic coefficients: evenly spaced in [a_star, a_dagger]
        self.a_n = np.linspace(a_star, a_dagger, K_trunc)
        # Generate K_trunc independent N(0,1) sequences
        xi = rng.standard_normal(size=(K_trunc, N_samples))
        # H_1 generators: a_n * xi_n
        self.sample_matrix = self.a_n[:, np.newaxis] * xi  # (K_trunc, N_samples)

    def analytical_bound(self, t: float) -> float:
        """
        Analytical upper bound on sup_n P(|a_n xi_n| >= t) via Markov inequality.

        P(|a_n xi_n| >= t) <= a_dagger^2 / t^2  (since E[xi_n^2] = 1)

        Parameters
        ----------
        t : float
            Threshold.

        Returns
        -------
        float
        """
        return self.a_dagger ** 2 / (t ** 2) if t > 0 else 1.0

    def build_checker(self) -> SmallConeDefinition:
        notes = (
            "Example 5.1: H_1 = {a_n * xi_n} with xi_n ~ N(0,1) i.i.d., "
            f"a_n in [{self.a_star}, {self.a_dagger}], truncated to K={self.K_trunc}. "
            "Bounded in L^0 via sup P(|h|>=t) <= a_dagger^2/t^2 -> 0 (Chebyshev). "
            "Closure in L^0: each individual variable is L^2, and the closure of "
            "the convex hull is compact in L^2 and hence in L^0 by inclusion. "
            "Zero excluded: a_n >= a_star > 0 and xi_n are nondegenerate."
        )
        return SmallConeDefinition(
            self.sample_matrix,
            name="Example 5.1 (L^2-bounded)",
            is_truncated=True,
            analytical_notes=notes,
        )


# ---------------------------------------------------------------------------
# Example 5.2 — Cauchy-based generator family
# ---------------------------------------------------------------------------

class GeneratorFamilyExample52:
    """
    Example 5.2: Independent standard Cauchy generators with bounded coefficients.

    zeta_n ~ Cauchy(0,1) i.i.d.
    c_n in [c_star, c_dagger] with 0 < c_star <= c_dagger.

    H_2 = { c_n * zeta_n : n = 1, 2, ..., K_trunc }.

    Boundedness in L^0:
      P(|c_n * zeta_n| >= t) = P(|zeta_n| >= t/c_n)
                              <= P(|zeta_1| >= t/c_dagger)  (scale monotone in c)
                              = 2/pi * arctan(c_dagger/t)
      This decays to 0 uniformly in n as t -> inf, even though E[|zeta_n|] = inf.

    This example shows that L^0-boundedness is STRICTLY WEAKER than L^1 or L^2
    boundedness, since Cauchy r.v.s have infinite first moments.

    Custom — paper Example 5.2.
    """

    def __init__(
        self,
        K_trunc: int = 50,
        N_samples: int = 100_000,
        c_star: float = 0.5,
        c_dagger: float = 2.0,
        rng_seed: int = 2,
    ) -> None:
        self.K_trunc = K_trunc
        self.N_samples = N_samples
        self.c_star = c_star
        self.c_dagger = c_dagger
        self.rng_seed = rng_seed

        rng = np.random.default_rng(rng_seed)
        self.c_n = np.linspace(c_star, c_dagger, K_trunc)
        zeta = rng.standard_cauchy(size=(K_trunc, N_samples))
        self.sample_matrix = self.c_n[:, np.newaxis] * zeta

    def analytical_bound(self, t: float) -> float:
        """
        Analytical upper bound: sup_n P(|c_n zeta_n| >= t) <= 2/pi * arctan(c_dag/t).
        """
        if t <= 0:
            return 1.0
        return (2.0 / np.pi) * np.arctan(self.c_dagger / t)

    def build_checker(self) -> SmallConeDefinition:
        notes = (
            "Example 5.2: H_2 = {c_n * zeta_n} with zeta_n ~ Cauchy(0,1) i.i.d., "
            f"c_n in [{self.c_star}, {self.c_dagger}], truncated to K={self.K_trunc}. "
            "Bounded in L^0 via sup P(|h|>=t) <= 2/pi * arctan(c_dag/t) -> 0 as t->inf. "
            "KEY INSIGHT: L^0-boundedness holds despite E[|zeta_n|] = infinity. "
            "This example validates that small-cone structure is weaker than L^1/L^2 "
            "boundedness and hence applies to heavy-tailed payoff families. "
            "Zero excluded: c_n >= c_star > 0 and Cauchy r.v.s are nondegenerate."
        )
        return SmallConeDefinition(
            self.sample_matrix,
            name="Example 5.2 (Cauchy, L^0-bounded only)",
            is_truncated=True,
            analytical_notes=notes,
        )


# ---------------------------------------------------------------------------
# Example 5.3 — Finite-dimensional generators
# ---------------------------------------------------------------------------

class GeneratorFamilyExample53:
    """
    Example 5.3: Finite family of linearly independent generators.

    A finite set of K_fin linearly independent random variables on a finite
    state space Omega = {omega_1, ..., omega_S}.

    Each generator g_k is a vector in R^S with P-almost-sure distinct values.
    The family H_3 = { g_1, ..., g_K_fin }.

    In finite dimension:
      - Convexity and closedness are exact (H is a finite set => conv(H) is a
        polytope, which is closed).
      - Boundedness in L^0 is automatic since max |g_k| < inf.
      - Zero excluded: verified by checking g_k != 0 for all k.

    Custom — paper Example 5.3.
    """

    def __init__(
        self,
        S: int = 10,
        K_fin: int = 4,
        N_mc_samples: int = 100_000,
        rng_seed: int = 3,
    ) -> None:
        self.S = S
        self.K_fin = K_fin
        self.rng_seed = rng_seed
        rng = np.random.default_rng(rng_seed)

        # State probabilities (uniform for simplicity)
        self.p = np.ones(S) / S

        # Generate K_fin linearly independent payoff vectors on S states
        # Using Gram-Schmidt to ensure independence
        raw = rng.standard_normal(size=(K_fin, S))
        # Simple check: if Gram-Schmidt is needed, apply it
        self.generators = self._gram_schmidt(raw)

        # Build sample matrix by sampling state indices
        # Monte Carlo: draw states with probabilities p
        state_indices = rng.choice(S, size=N_mc_samples, p=self.p)
        self.sample_matrix = self.generators[:, state_indices]  # (K_fin, N_mc)

    @staticmethod
    def _gram_schmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Orthonormalize rows of a matrix using Gram-Schmidt.

        Parameters
        ----------
        vectors : np.ndarray, shape (K, S)

        Returns
        -------
        np.ndarray, shape (K, S)
        """
        basis = []
        for v in vectors:
            w = v.copy()
            for b in basis:
                w -= np.dot(w, b) * b
            norm = np.linalg.norm(w)
            if norm > 1e-10:
                basis.append(w / norm)
        return np.array(basis)

    def check_linear_independence(self) -> bool:
        """Verify linear independence of the generator payoff vectors."""
        rank = np.linalg.matrix_rank(self.generators)
        return bool(rank == self.K_fin)

    def check_convex_hull_properties(self) -> dict:
        """
        Compute properties of conv(H_3) using exact arithmetic on state payoffs.

        Returns
        -------
        dict with 'n_vertices', 'diameter', 'zero_in_conv'.
        """
        from scipy.spatial import ConvexHull

        # Project generators to 2D for visualisation if K_fin > 2
        G = self.generators  # (K_fin, S)
        n_verts = self.K_fin
        # Check if 0 in conv(H_3) using LP
        zero_in_conv = self._zero_in_convex_hull(G)
        diameter = float(np.max(np.linalg.norm(G[:, np.newaxis, :] - G[np.newaxis, :, :], axis=-1)))
        return {
            "n_vertices": n_verts,
            "linear_independent": self.check_linear_independence(),
            "diameter": diameter,
            "zero_in_conv_hull": zero_in_conv,
            "all_nonzero": bool(np.all(np.linalg.norm(G, axis=1) > 1e-10)),
        }

    @staticmethod
    def _zero_in_convex_hull(G: np.ndarray) -> bool:
        """
        Check if 0 in conv({g_1,...,g_K}) using LP.

        min 0 s.t. sum_k lambda_k g_k = 0, lambda >= 0, sum lambda = 1.
        """
        from scipy.optimize import linprog

        K, S = G.shape
        # Variables: lambda (K,)
        # sum_k lambda_k g_{k,s} = 0 for all s
        # sum_k lambda_k = 1
        # lambda_k >= 0
        A_eq_1 = G.T                   # (S, K)  -- payoff = 0
        b_eq_1 = np.zeros(S)
        A_eq_2 = np.ones((1, K))       # sum = 1
        b_eq_2 = np.array([1.0])
        A_eq = np.vstack([A_eq_1, A_eq_2])
        b_eq = np.concatenate([b_eq_1, b_eq_2])
        bounds = [(0, None)] * K
        c = np.zeros(K)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return res.success and np.isclose(np.dot(G.T, res.x), 0, atol=1e-6).all()

    def build_checker(self) -> SmallConeDefinition:
        conv_props = self.check_convex_hull_properties()
        notes = (
            f"Example 5.3: H_3 = finite family of {self.K_fin} linearly independent "
            f"generators on S={self.S} states (uniform p). "
            "SC1: conv(H_3) is a polytope (closed and convex). "
            "SC2: Finite family => trivially closed in L^0. "
            "SC3: Bounded in L^0 since max |g_k| < inf on finite omega. "
            f"SC4: All generators nonzero = {conv_props['all_nonzero']}. "
            f"Zero in conv(H_3) = {conv_props['zero_in_conv_hull']}. "
            "(For SC4 we need 0 not in H_3, which holds since generators are nonzero.)"
        )
        return SmallConeDefinition(
            self.sample_matrix,
            name="Example 5.3 (Finite-dim)",
            is_truncated=False,
            analytical_notes=notes,
        )


# ---------------------------------------------------------------------------
# Lemma 2.4 illustration
# ---------------------------------------------------------------------------

class Lemma24Illustration:
    """
    Illustrates Lemma 2.4: if C is a closed cone, L is a small cone, and
    C ∩ (-L) = {0}, then C + L is closed in L^0.

    This is demonstrated in a finite-state model where the statement is
    exact (finite-dimensional closed cones are closed in all topologies).

    C  = dynamic gain cone: K_0 = {phi*(x1 - x0) : phi in R}
    L  = cone generated by a finite small-cone generator set.

    Custom — paper Lemma 2.4.
    """

    def __init__(
        self,
        model_x1: np.ndarray,
        x0: float,
        p: np.ndarray,
        small_cone_generators: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        model_x1 : np.ndarray, shape (S,)
            Terminal asset values.
        x0 : float
            Initial asset value.
        p : np.ndarray, shape (S,)
            State probabilities.
        small_cone_generators : np.ndarray, shape (K, S)
            Generator payoff vectors for the small cone.
        """
        self.x1 = model_x1
        self.x0 = x0
        self.p = p
        self.generators = small_cone_generators
        self.S = len(p)
        self.gain_vec = model_x1 - x0

    def check_nonredundancy(self) -> dict:
        """
        Check C ∩ (-L) = {0}.

        This means: there is no nonzero element that is simultaneously
        in the dynamic-gain cone K_0 and in -L (= {-lambda*h : lambda>=0, h in H}).

        In finite state, this becomes: no phi and lambda_k>=0 (not all zero)
        such that phi*gain = -sum_k lambda_k * g_k  and  phi*gain != 0.

        Parameters
        ----------
        None

        Returns
        -------
        dict
        """
        from scipy.optimize import linprog

        K = len(self.generators)
        S = self.S

        # Check if phi*gain + sum_k lambda_k * g_k = 0 has nontrivial solution
        # Variables: [phi_plus, phi_minus, lambda_1,...,lambda_K]
        # phi = phi_plus - phi_minus, all nonneg
        # (phi_plus - phi_minus)*gain + G^T lambda = 0
        A_eq = np.zeros((S, 2 + K))
        A_eq[:, 0] = self.gain_vec       # phi_plus
        A_eq[:, 1] = -self.gain_vec      # phi_minus
        for k in range(K):
            A_eq[:, 2 + k] = self.generators[k]  # lambda_k

        b_eq = np.zeros(S)
        # Objective: maximize sum of variables (to find nontrivial solution)
        c = -np.ones(2 + K)
        bounds = [(0, None)] * (2 + K)
        # Add constraint sum >= epsilon to force nontrivial
        # A_ub: -sum <= -eps
        A_ub = -np.ones((1, 2 + K))
        b_ub = np.array([-1e-6])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if res.success:
            # A nontrivial element exists => redundancy
            return {
                "nonredundancy": False,
                "C_cap_neg_L_eq_zero": False,
                "note": "Found nonzero element in K_0 ∩ (-L) — redundancy present.",
            }
        else:
            return {
                "nonredundancy": True,
                "C_cap_neg_L_eq_zero": True,
                "note": "K_0 ∩ (-L) = {0} — small cone is nonredundant w.r.t. dynamic trading.",
            }

    def check_closedness_C_plus_L(self) -> dict:
        """
        In finite state, C + L is automatically closed (finite-dimensional cones
        generated by finitely many vectors are closed).

        Returns
        -------
        dict with analytical justification.
        """
        return {
            "C_plus_L_closed": True,
            "justification": (
                "In a finite-state (finite-dimensional) setting, any finitely-generated "
                "cone C + L is closed in R^S (which is L^0 of the finite probability space). "
                "This directly illustrates Lemma 2.4's conclusion under the assumption "
                "that C ∩ (-L) = {0}."
            ),
        }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(N_samples: int = 100_000) -> dict:
    """
    Run the full Exp 2 small-cone validation pipeline.

    Parameters
    ----------
    N_samples : int
        Monte Carlo samples for tail probability estimation.

    Returns
    -------
    dict
        Full validation report for Examples 5.1, 5.2, 5.3 and Lemma 2.4.
    """
    results = {}

    # Example 5.1
    ex51 = GeneratorFamilyExample51(K_trunc=50, N_samples=N_samples)
    checker51 = ex51.build_checker()
    results["example_5_1"] = checker51.validate()
    results["example_5_1"]["analytical_bound_at_10"] = ex51.analytical_bound(10.0)
    results["example_5_1"]["analytical_bound_at_50"] = ex51.analytical_bound(50.0)

    # Example 5.2
    ex52 = GeneratorFamilyExample52(K_trunc=50, N_samples=N_samples)
    checker52 = ex52.build_checker()
    results["example_5_2"] = checker52.validate()
    results["example_5_2"]["analytical_bound_at_10"] = ex52.analytical_bound(10.0)
    results["example_5_2"]["analytical_bound_at_50"] = ex52.analytical_bound(50.0)

    # Example 5.3
    ex53 = GeneratorFamilyExample53(S=10, K_fin=4, N_mc_samples=N_samples)
    checker53 = ex53.build_checker()
    conv_props = ex53.check_convex_hull_properties()
    results["example_5_3"] = checker53.validate()
    results["example_5_3"]["convex_hull_properties"] = conv_props

    # Lemma 2.4 illustration using Example 5.3 generators in a finite-state model
    S = 10
    rng = np.random.default_rng(99)
    p = np.ones(S) / S
    x1 = np.sort(rng.uniform(0.5, 1.5, size=S))
    x0 = float(np.mean(x1))  # fair initial price
    lemma24 = Lemma24Illustration(
        model_x1=x1,
        x0=x0,
        p=p,
        small_cone_generators=ex53.generators,
    )
    results["lemma_2_4"] = {
        "nonredundancy": lemma24.check_nonredundancy(),
        "closedness": lemma24.check_closedness_C_plus_L(),
    }

    return results


def plot_l0_boundedness_comparison(
    N_samples: int = 100_000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot L^0 tail probabilities for Examples 5.1 and 5.2 alongside analytical bounds.

    Parameters
    ----------
    N_samples : int
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    ex51 = GeneratorFamilyExample51(K_trunc=50, N_samples=N_samples)
    ex52 = GeneratorFamilyExample52(K_trunc=50, N_samples=N_samples)

    thresholds = np.logspace(0.2, 2.5, 40)

    sup51 = [sup_tail_probability(ex51.sample_matrix, t) for t in thresholds]
    sup52 = [sup_tail_probability(ex52.sample_matrix, t) for t in thresholds]
    bound51 = [ex51.analytical_bound(t) for t in thresholds]
    bound52 = [ex52.analytical_bound(t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(thresholds, sup51, "o-", color="steelblue", label="Ex 5.1 empirical (L²-bounded)")
    ax.loglog(thresholds, bound51, "--", color="steelblue", label="Ex 5.1 Chebyshev bound")
    ax.loglog(thresholds, sup52, "s-", color="crimson", label="Ex 5.2 empirical (Cauchy, L⁰ only)")
    ax.loglog(thresholds, bound52, "--", color="crimson", label="Ex 5.2 analytical bound")
    ax.set_xlabel("Threshold n")
    ax.set_ylabel("sup_H P(|h| ≥ n)")
    ax.set_title("Exp 2 — L⁰ Boundedness: Examples 5.1 and 5.2\n"
                 "(both decay → 0 confirming small-cone property SC3)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
